import math
from collections.abc import Mapping
from typing import Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# from flash_attn.ops.fused_dense import FusedMLP, FusedDense
from huggingface_hub import PyTorchModelHubMixin

from ..components import (
    Rotary,
    apply_rotary_pos_emb,
    bias_dropout_add_scale_fused_inference,
    bias_dropout_add_scale_fused_train,
    modulate_fused,
)
from .config import SEDDConfig


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


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
    return torch.addmm(x_skip.view(-1, dim_out), x.view(-1, dim_in), W.T, alpha=residual_scale).view(
        *x.shape[:-1], dim_out
    )


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
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
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
        return bias_dropout_add_scale_fused_train if self.training else bias_dropout_add_scale_fused_inference

    def forward(self, x, rotary_cos_sin, c, seqlens=None):
        from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func

        batch_size, seq_len = x.shape[0], x.shape[1]

        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c)[:, None].chunk(
            6, dim=2
        )

        # attention operation
        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
        # dtype0 = x.dtype

        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=self.n_heads)
        with torch.amp.autocast(device_type="cuda", enabled=False):
            cos, sin = rotary_cos_sin
            qkv = apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
        qkv = rearrange(qkv, "b s ... -> (b s) ...")
        if seqlens is None:
            cu_seqlens = torch.arange(
                0, (batch_size + 1) * seq_len, step=seq_len, dtype=torch.int32, device=qkv.device
            )
        else:
            cu_seqlens = seqlens.cumsum(-1)
        x = flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, seq_len, 0.0, causal=False)

        x = rearrange(x, "(b s) h d -> b s (h d)", b=batch_size)

        x = bias_dropout_scale_fn(self.attn_out(x), None, gate_msa, x_skip, self.dropout)

        # mlp operation
        x = bias_dropout_scale_fn(
            self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)), None, gate_mlp, x, self.dropout
        )
        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        """
        Mode arg: 0 -> use a learned layer, 1 -> use eigenvectors,
        2-> add in eigenvectors, 3 -> use pretrained embedding matrix
        """
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
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
    config_class = SEDDConfig

    def __init__(self, config: Union[SEDDConfig, Mapping[str, Any], Any]):
        super().__init__()

        if not isinstance(config, SEDDConfig):
            config = self._standardize_config(config)

        self.config: SEDDConfig = config

        self.absorb = config.graph_type == "absorb"
        vocab_size = config.vocab_size

        self.vocab_embed = EmbeddingLayer(config.hidden_size, vocab_size)
        self.sigma_map = TimestepEmbedder(config.cond_dim)
        self.rotary_emb = Rotary(config.hidden_size // config.n_heads)

        self.blocks = nn.ModuleList(
            [
                DDiTBlock(
                    config.hidden_size,
                    config.n_heads,
                    config.cond_dim,
                    dropout=config.dropout,
                )
                for _ in range(config.n_blocks)
            ]
        )

        self.output_layer = DDitFinalLayer(config.hidden_size, vocab_size, config.cond_dim)
        self.scale_by_sigma = config.scale_by_sigma

    def _get_bias_dropout_scale(self):
        return bias_dropout_add_scale_fused_train if self.training else bias_dropout_add_scale_fused_inference

    def forward(self, indices, sigma):
        x = self.vocab_embed(indices)
        c = F.silu(self.sigma_map(sigma))

        rotary_cos_sin = self.rotary_emb(x)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None)

            x = self.output_layer(x, c)

        if self.scale_by_sigma:
            assert self.absorb, "Haven't configured this to work."
            esigm1_log = torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1).log().to(x.dtype)[:, None, None]
            x = x - esigm1_log - np.log(x.shape[-1] - 1)  # this will be approximately averaged at 0

        x = torch.scatter(x, -1, indices[..., None], torch.zeros_like(x[..., :1]))

        return x

    @staticmethod
    def _standardize_config(config: Union[Mapping[str, Any], Any]) -> SEDDConfig:
        def _maybe_get(obj, key, default=None):
            if isinstance(obj, Mapping):
                return obj[key] if key in obj else default
            return getattr(obj, key, default)

        tokens = _maybe_get(config, "tokens")
        if tokens is None:
            raise ValueError("`tokens` must be provided in the SEDD configuration.")

        graph_cfg = _maybe_get(config, "graph", {})
        graph_type = _maybe_get(graph_cfg, "type", "absorb")

        model_cfg = _maybe_get(config, "model", {})
        hidden_size = _maybe_get(model_cfg, "hidden_size")
        cond_dim = _maybe_get(model_cfg, "cond_dim")
        length = _maybe_get(model_cfg, "length")
        n_blocks = _maybe_get(model_cfg, "n_blocks")
        n_heads = _maybe_get(model_cfg, "n_heads")
        scale_by_sigma = _maybe_get(model_cfg, "scale_by_sigma", True)
        dropout = _maybe_get(model_cfg, "dropout", 0.1)

        missing = [
            name
            for name, value in {
                "hidden_size": hidden_size,
                "cond_dim": cond_dim,
                "length": length,
                "n_blocks": n_blocks,
                "n_heads": n_heads,
            }.items()
            if value is None
        ]
        if missing:
            raise ValueError(f"Missing model hyperparameters in configuration: {', '.join(missing)}")

        return SEDDConfig(
            tokens=int(tokens),
            graph_type=str(graph_type),
            hidden_size=int(hidden_size),
            cond_dim=int(cond_dim),
            length=int(length),
            n_blocks=int(n_blocks),
            n_heads=int(n_heads),
            scale_by_sigma=bool(scale_by_sigma),
            dropout=float(dropout),
        )
