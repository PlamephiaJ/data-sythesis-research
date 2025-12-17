import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# from flash_attn.ops.fused_dense import FusedMLP, FusedDense
from huggingface_hub import PyTorchModelHubMixin
from omegaconf import OmegaConf

from kernel.fused_add_dropout_scale import (
    bias_dropout_add_scale_fused_inference,
    bias_dropout_add_scale_fused_train,
    modulate_fused,
)

from . import rotary
from .caption_encoder import CaptionEncoder


# def modulate(x, shift, scale):
#     return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        with torch.amp.autocast(device_type="cuda", enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


def residual_linear(x, W, x_skip, residual_scale):
    """x_skip + residual_scale * W @ x"""
    dim_out, dim_in = W.shape[0], W.shape[1]
    return torch.addmm(
        x_skip.view(-1, dim_out), x.view(-1, dim_in), W.T, alpha=residual_scale
    ).view(*x.shape[:-1], dim_out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, silu=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, cond_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
        self.num_classes = num_classes

        # TODO think of initializing with 0.02 std deviation like in original DiT paper

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core Model                                    #
#################################################################################


class DDiTBlock(nn.Module):

    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )
        self.dropout2 = nn.Dropout(dropout)

        self.dropout = dropout

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )

    def forward(self, x, rotary_cos_sin, c, seqlens=None, attention_mask=None):
        """
        Docstring for forward

        :param x: (batch, seq_len, dim), input tensor
        :param rotary_cos_sin: (cos, sin) tuple for rotary positional embeddings
        :param c: (batch, cond_dim), conditioning tensor
        :param seqlens: (batch,) sequence lengths for variable length sequences
        """
        from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func

        batch_size, max_seq_len = x.shape[0], x.shape[1]

        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        # 将 conditioning c 映射到 shift 和 scale 参数
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
        )

        # attention operation
        x_skip = x
        x = self.norm1(x)
        x = modulate_fused(x, shift_msa, scale_msa)
        # dtype0 = x.dtype

        qkv = self.attn_qkv(x)
        qkv = rearrange(
            qkv, "b s (three h d) -> b s three h d", three=3, h=self.n_heads
        )
        with torch.amp.autocast(device_type="cuda", enabled=False):
            cos, sin = rotary_cos_sin
            qkv = rotary.apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
        qkv = rearrange(qkv, "b s ... -> (b s) ...")
        if seqlens is None:
            # cumulative sequence lengths. Required by flash attention for variable length sequences
            cu_seqlens = torch.arange(
                0,
                (batch_size + 1) * max_seq_len,
                step=max_seq_len,
                dtype=torch.int32,
                device=qkv.device,
            ).to(torch.int32)
        else:
            cu_seqlens = seqlens.cumsum(-1).to(torch.int32)  # cumulative sum of seqlens
        x = flash_attn_varlen_qkvpacked_func(
            qkv, cu_seqlens, max_seq_len, 0.0, causal=False
        )

        x = rearrange(x, "(b s) h d -> b s (h d)", b=batch_size)

        x = bias_dropout_scale_fn(
            self.attn_out(x), None, gate_msa, x_skip, self.dropout
        )

        # mlp operation
        x = bias_dropout_scale_fn(
            self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
            None,
            gate_mlp,
            x,
            self.dropout,
        )
        return x


class EmbeddingLayer(nn.Module):

    def __init__(self, dim, vocab_size):
        """
        Mode arg: 0 -> use a learned layer, 1 -> use eigenvectors,
        2-> add in eigenvectors, 3 -> use pretrained embedding matrix
        """
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_size, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]


class DDitFinalLayer(nn.Module):

    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate_fused(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SEDD(nn.Module, PyTorchModelHubMixin):

    def __init__(self, config):
        super().__init__()

        if isinstance(config, dict):
            config = OmegaConf.create(config)

        self.config = config

        self.absorb = config.graph.type == "absorb"
        vocab_size = config.tokens + (1 if self.absorb else 0)

        self.vocab_embed = EmbeddingLayer(config.model.hidden_size, vocab_size)
        self.sigma_map = TimestepEmbedder(config.model.cond_dim)
        self.rotary_emb = rotary.Rotary(
            config.model.hidden_size // config.model.n_heads
        )

        self.blocks = nn.ModuleList(
            [
                DDiTBlock(
                    config.model.hidden_size,
                    config.model.n_heads,
                    config.model.cond_dim,
                    dropout=config.model.dropout,
                )
                for _ in range(config.model.n_blocks)
            ]
        )

        self.output_layer = DDitFinalLayer(
            config.model.hidden_size, vocab_size, config.model.cond_dim
        )
        self.scale_by_sigma = config.model.scale_by_sigma

        self.caption_encoder = CaptionEncoder(
            name=config.model.caption_encoder.name,
            cond_dim=config.model.cond_dim,
            pool=config.model.caption_encoder.pool,
            dropout=config.model.caption_encoder.dropout,
            freeze=config.model.caption_encoder.freeze,
        )

    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )

    def forward(
        self,
        x_input_ids,
        x_attention_mask,
        caption_input_ids=None,
        caption_attention_mask=None,
        sigma=None,
    ):
        """
        Docstring for forward

        :param indices: (batch, seq_len), a batch of token indices
        :param sigma: (batch,), a batch of sigma values
        """
        # x: batch seq_len voval_size, transform indices to embeddings
        x = self.vocab_embed(x_input_ids)

        if caption_input_ids is not None and caption_attention_mask is not None:
            cond_style = self.caption_encoder(
                input_ids=caption_input_ids, attention_mask=caption_attention_mask
            )
        else:
            cond_style = self.caption_encoder(None, None)

        # cond_sigma: batch cond_dim, condition embedding from sigma
        cond_sigma = F.silu(self.sigma_map(sigma))

        c = cond_sigma + cond_style

        rotary_cos_sin = self.rotary_emb(x)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            seqlens = x_attention_mask.sum(dim=1).to(torch.int32)  # [B]
            for blk in self.blocks:
                x = blk(
                    x,
                    rotary_cos_sin,
                    c,
                    seqlens=seqlens,
                    attention_mask=x_attention_mask,
                )

            x = self.output_layer(x, c)

        if self.scale_by_sigma:
            assert self.absorb, "Haven't configured this to work."
            esigm1_log = (
                torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1)
                .log()
                .to(x.dtype)[:, None, None]
            )
            x = (
                x - esigm1_log - np.log(x.shape[-1] - 1)
            )  # this will be approximately averaged at 0

        # x: batch seq_len voval_size
        # indices: batch seq_len
        x = torch.scatter(
            x, dim=-1, index=x_input_ids[..., None], src=torch.zeros_like(x[..., :1])
        )

        return x
