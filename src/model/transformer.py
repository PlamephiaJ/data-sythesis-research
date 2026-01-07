# -*- coding: utf-8 -*-
"""
A configuration-driven refactor that supports:
  1) c = cond_sigma                      (sigma_only)
  2) c = cond_sigma + cond_style         (sigma_plus_style)
  3) cond_style as tokens for cross-attn (cross_attn_style)
and keeps forward / blocks extensible via Registry + Strategy + optional adapters.

This file is intended as a drop-in replacement for the relevant parts of your current code.
You will need ONE small change in CaptionEncoder:
  - add return_tokens=True support (see notes at bottom). If you cannot change it now,
    this code will still run but cross-attn mode will raise a clear error.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin
from omegaconf import OmegaConf

from kernel.fused_add_dropout_scale import (
    bias_dropout_add_scale_fused_inference,
    bias_dropout_add_scale_fused_train,
    modulate_fused,
)

from . import rotary
from .encoder import CaptionEncoder


#################################################################################
#                                  Layers                                       #
#################################################################################


class LayerNorm(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(device_type="cuda", enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor, dim: int, max_period: int = 10000
    ) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class EmbeddingLayer(nn.Module):

    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_size, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding[x]


#################################################################################
#                              Forward Context                                  #
#################################################################################


@dataclass
class ForwardContext:
    sigma: torch.Tensor  # (B,)
    cond_sigma: torch.Tensor  # (B, cond_dim)
    c: torch.Tensor  # (B, cond_dim) for adaLN

    rotary_cos_sin: Tuple[torch.Tensor, torch.Tensor]

    x_attention_mask: torch.Tensor  # (B, S) 1 valid, 0 pad
    seqlens: torch.Tensor  # (B,) int32

    cond_style: Optional[torch.Tensor] = None  # (B, cond_dim) pooled
    style_tokens: Optional[torch.Tensor] = None  # (B, Lc, dim) for cross-attn KV
    style_attention_mask: Optional[torch.Tensor] = None  # (B, Lc)


#################################################################################
#                       Conditioning Strategy                                   #
#################################################################################

_COND_REGISTRY: Dict[str, Any] = {}


def register_conditioning(name: str):
    def deco(cls):
        _COND_REGISTRY[name] = cls
        return cls

    return deco


class ConditioningStrategy:
    """
    Build the conditioning pack:
      - c: (B, cond_dim) for adaLN
      - cond_style: optional pooled style (B, cond_dim)
      - style_tokens/style_attention_mask: optional for cross-attn
    """

    def build(
        self,
        *,
        cond_sigma: torch.Tensor,
        caption_encoder: CaptionEncoder,
        caption_input_ids: Optional[torch.Tensor],
        caption_attention_mask: Optional[torch.Tensor],
    ) -> Dict[str, Optional[torch.Tensor]]:
        raise NotImplementedError


@register_conditioning("sigma_only")
class SigmaOnly(ConditioningStrategy):

    def build(
        self, *, cond_sigma, caption_encoder, caption_input_ids, caption_attention_mask
    ):
        return {
            "c": cond_sigma,
            "cond_style": None,
            "style_tokens": None,
            "style_attention_mask": None,
        }


@register_conditioning("sigma_plus_style")
class SigmaPlusStyle(ConditioningStrategy):

    def build(
        self, *, cond_sigma, caption_encoder, caption_input_ids, caption_attention_mask
    ):
        if caption_input_ids is not None and caption_attention_mask is not None:
            cond_style = caption_encoder(
                input_ids=caption_input_ids, attention_mask=caption_attention_mask
            )
        else:
            cond_style = caption_encoder(None, None)
        return {
            "c": cond_sigma + cond_style,
            "cond_style": cond_style,
            "style_tokens": None,
            "style_attention_mask": None,
        }


@register_conditioning("cross_attn_style")
class CrossAttnStyle(ConditioningStrategy):
    """
    - adaLN uses only cond_sigma (stable and minimal)
    - style goes as tokens into DDiTBlock cross-attn
    """

    def build(
        self, *, cond_sigma, caption_encoder, caption_input_ids, caption_attention_mask
    ):
        # EXPECT: caption_encoder(..., return_tokens=True) -> (pooled, tokens)
        try:
            pooled, tokens, token_mask = caption_encoder(
                input_ids=caption_input_ids,
                attention_mask=caption_attention_mask,
                return_tokens=True,
            )
        except TypeError as e:
            raise TypeError(
                "CaptionEncoder must support return_tokens=True for cross-attn mode. "
                "Please implement it (see notes at bottom)."
            ) from e

        return {
            "c": cond_sigma,
            "cond_style": pooled,
            "style_tokens": tokens,
            "style_attention_mask": token_mask,
        }


def build_conditioning_strategy(cfg: Any) -> ConditioningStrategy:
    # cfg can be missing: default sigma_plus_style
    mode = "sigma_plus_style"
    if cfg is not None and "mode" in cfg:
        mode = str(cfg.mode)
    if mode not in _COND_REGISTRY:
        raise KeyError(
            f"Unknown conditioning.mode={mode}. Supported: {list(_COND_REGISTRY.keys())}"
        )
    return _COND_REGISTRY[mode]()


#################################################################################
#                         Optional Cross-Attention Adapter                      #
#################################################################################


class CrossAttnAdapter(nn.Module):
    """
    FlashAttention varlen cross-attn adapter (Q from x, KV from style_tokens).
    Assumes:
      - x: (B, S, dim)
      - style_tokens: (B, L, dim)  (if not dim, add projection before passing in)
      - masks: (B, S)/(B, L) with 1 valid, 0 pad
    """

    def __init__(self, dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert dim % n_heads == 0, "dim must be divisible by n_heads"

        self.norm = LayerNorm(dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, 2 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        *,
        x_attention_mask: torch.Tensor,
        style_tokens: torch.Tensor,
        style_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        from flash_attn.bert_padding import pad_input, unpad_input
        from flash_attn.flash_attn_interface import flash_attn_varlen_kvpacked_func

        B, S, _ = x.shape

        # Normalize + project Q
        xq = self.norm(x)
        q = self.q(xq)  # (B,S,dim)
        q = rearrange(q, "b s (h d) -> b s h d", h=self.n_heads)

        # Project KV from style tokens
        kv = self.kv(style_tokens)  # (B,L,2*dim)
        kv = rearrange(kv, "b l (two h d) -> b l two h d", two=2, h=self.n_heads)

        # Unpad Q
        q_2d = rearrange(q, "b s h d -> b s (h d)")
        q_unpad, q_indices, q_cu_seqlens, q_max_seqlen, _ = unpad_input(
            q_2d, x_attention_mask
        )
        q_unpad = rearrange(q_unpad, "t (h d) -> t h d", h=self.n_heads)

        # Unpad KV (packed)
        kv_2d = rearrange(kv, "b l two h d -> b l (two h d)")
        kv_unpad, _, kv_cu_seqlens, kv_max_seqlen, _ = unpad_input(
            kv_2d, style_attention_mask
        )
        kv_unpad = rearrange(
            kv_unpad, "t (two h d) -> t two h d", two=2, h=self.n_heads
        )

        # Cross-attn (varlen)
        out_unpad = flash_attn_varlen_kvpacked_func(
            q_unpad,
            kv_unpad,
            q_cu_seqlens.to(torch.int32),
            kv_cu_seqlens.to(torch.int32),
            q_max_seqlen,
            kv_max_seqlen,
            dropout_p=0.0,
            causal=False,
        )  # (Tq, H, D)

        out_unpad = rearrange(out_unpad, "t h d -> t (h d)")
        out = pad_input(out_unpad, q_indices, B, S)  # (B,S,dim)
        return self.out(out)


#################################################################################
#                                 Core Blocks                                   #
#################################################################################

_BLOCK_REGISTRY: Dict[str, Any] = {}


def register_block(name: str):
    def deco(cls):
        _BLOCK_REGISTRY[name] = cls
        return cls

    return deco


def build_block(
    block_cfg: Any, *, dim: int, n_heads: int, cond_dim: int, dropout: float
) -> nn.Module:
    btype = "ddit_flash"
    enable_cross_attn = False
    if block_cfg is not None:
        if "type" in block_cfg:
            btype = str(block_cfg.type)
        if "enable_cross_attn" in block_cfg:
            enable_cross_attn = bool(block_cfg.enable_cross_attn)

    if btype not in _BLOCK_REGISTRY:
        raise KeyError(
            f"Unknown block type={btype}. Supported: {list(_BLOCK_REGISTRY.keys())}"
        )

    # Pass through additional kwargs (if any)
    extra = {}
    if block_cfg is not None:
        for k, v in block_cfg.items():
            if k in ("type", "enable_cross_attn"):
                continue
            extra[k] = v

    return _BLOCK_REGISTRY[btype](
        dim=dim,
        n_heads=n_heads,
        cond_dim=cond_dim,
        dropout=dropout,
        enable_cross_attn=enable_cross_attn,
        **extra,
    )


@register_block("ddit_flash")
class DDiTBlock(nn.Module):
    """
    Self-attn + (optional cross-attn) + MLP with adaLN.
    Self-attn path uses your FlashAttention varlen qkvpacked implementation.
    Cross-attn is optional via CrossAttnAdapter, driven by config and ctx.style_tokens.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        cond_dim: int,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        enable_cross_attn: bool = False,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim

        # Self-attention
        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout = float(dropout)

        # MLP
        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )

        # adaLN modulation
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

        # Optional cross-attn adapter + gate
        self.cross_attn = (
            CrossAttnAdapter(dim, n_heads, dropout) if enable_cross_attn else None
        )
        self.xattn_gate = (
            nn.Linear(cond_dim, dim, bias=True) if enable_cross_attn else None
        )
        if self.xattn_gate is not None:
            self.xattn_gate.weight.data.zero_()
            self.xattn_gate.bias.data.zero_()

    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )

    def forward(
        self,
        x: torch.Tensor,  # (B,S,dim)
        rotary_cos_sin: Tuple[torch.Tensor, torch.Tensor],
        c: torch.Tensor,  # (B,cond_dim)
        *,
        seqlens: Optional[torch.Tensor] = None,  # (B,) int32
        attention_mask: Optional[torch.Tensor] = None,  # (B,S) 1 valid 0 pad
        style_tokens: Optional[torch.Tensor] = None,  # (B,L,dim)
        style_attention_mask: Optional[torch.Tensor] = None,  # (B,L)
    ) -> torch.Tensor:
        from flash_attn.bert_padding import pad_input, unpad_input
        from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func

        B, S = x.shape[0], x.shape[1]
        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        # adaLN params
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
        )

        # -------------------------
        # 1) Self-Attention (FlashAttn varlen)
        # -------------------------
        x_skip = x
        x1 = self.norm1(x)
        x1 = modulate_fused(x1, shift_msa, scale_msa)

        qkv = self.attn_qkv(x1)  # (B,S,3*dim)
        qkv = rearrange(
            qkv, "b s (three h d) -> b s three h d", three=3, h=self.n_heads
        )

        # rotary
        with torch.amp.autocast(device_type="cuda", enabled=False):
            cos, sin = rotary_cos_sin
            qkv = rotary.apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))

        if attention_mask is None:
            # If you truly have no mask, create one (all valid). But your model uses masks, so keep strict.
            attention_mask = torch.ones((B, S), device=qkv.device, dtype=torch.int32)

        qkv_2d = rearrange(qkv, "b s three h d -> b s (three h d)")
        qkv_unpad, indices, cu_seqlens_u, max_seqlen, _ = unpad_input(
            qkv_2d, attention_mask
        )
        qkv_unpad = rearrange(
            qkv_unpad, "t (three h d) -> t three h d", three=3, h=self.n_heads
        )

        out_unpad = flash_attn_varlen_qkvpacked_func(
            qkv_unpad,
            cu_seqlens_u.to(torch.int32),
            max_seqlen,
            dropout_p=0.0,
            causal=False,
        )  # (T, H, D)

        out_unpad = rearrange(out_unpad, "t h d -> t (h d)")
        attn_out = pad_input(out_unpad, indices, B, S)  # (B,S,dim)

        x = bias_dropout_scale_fn(
            self.attn_out(attn_out), None, gate_msa, x_skip, self.dropout
        )

        # -------------------------
        # 2) Optional Cross-Attention (style tokens)
        # -------------------------
        if (
            self.cross_attn is not None
            and style_tokens is not None
            and style_attention_mask is not None
        ):
            x_skip = x
            x_xattn = self.cross_attn(
                x,
                x_attention_mask=attention_mask,
                style_tokens=style_tokens,
                style_attention_mask=style_attention_mask,
            )
            gate_x = self.xattn_gate(c)[:, None, :]  # (B,1,dim)
            x = bias_dropout_scale_fn(x_xattn, None, gate_x, x_skip, self.dropout)

        # -------------------------
        # 3) MLP
        # -------------------------
        x = bias_dropout_scale_fn(
            self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
            None,
            gate_mlp,
            x,
            self.dropout,
        )
        return x


#################################################################################
#                                 Final Layer                                   #
#################################################################################


class DDitFinalLayer(nn.Module):

    def __init__(self, hidden_size: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate_fused(self.norm_final(x), shift, scale)
        return self.linear(x)


#################################################################################
#                                  SEDD Model                                   #
#################################################################################


class SEDD(nn.Module, PyTorchModelHubMixin):

    def __init__(self, config: Any):
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

        # caption encoder stays as-is; cross-attn mode requires return_tokens=True support.
        self.caption_encoder = CaptionEncoder(
            name=config.model.caption_encoder.name,
            cond_dim=config.model.cond_dim,
            pool=config.model.caption_encoder.pool,
            dropout=config.model.caption_encoder.dropout,
            freeze=config.model.caption_encoder.freeze,
            token_dim=config.model.hidden_size,
        )

        # conditioning strategy (config-driven)
        cond_cfg = config.model.conditioning if "conditioning" in config.model else None
        self.cond_strategy = build_conditioning_strategy(cond_cfg)

        # blocks (config-driven)
        block_cfg = config.model.blocks if "blocks" in config.model else None
        self.blocks = nn.ModuleList(
            [
                build_block(
                    block_cfg,
                    dim=config.model.hidden_size,
                    n_heads=config.model.n_heads,
                    cond_dim=config.model.cond_dim,
                    dropout=float(config.model.dropout),
                )
                for _ in range(config.model.n_blocks)
            ]
        )

        self.output_layer = DDitFinalLayer(
            config.model.hidden_size, vocab_size, config.model.cond_dim
        )
        self.scale_by_sigma = bool(config.model.scale_by_sigma)

    def forward(
        self,
        x_input_ids: torch.Tensor,
        x_attention_mask: torch.Tensor,
        caption_input_ids: Optional[torch.Tensor] = None,
        caption_attention_mask: Optional[torch.Tensor] = None,
        sigma: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # embeddings
        x = self.vocab_embed(x_input_ids)

        # sigma conditioning
        cond_sigma = F.silu(self.sigma_map(sigma))

        # rotary
        rotary_cos_sin = self.rotary_emb(x)

        # mask-derived lengths (used by flash-attn unpad)
        seqlens = x_attention_mask.sum(dim=1).to(torch.int32)  # (B,)

        # build conditioning pack
        cond_pack = self.cond_strategy.build(
            cond_sigma=cond_sigma,
            caption_encoder=self.caption_encoder,
            caption_input_ids=caption_input_ids,
            caption_attention_mask=caption_attention_mask,
        )

        ctx = ForwardContext(
            sigma=sigma,
            cond_sigma=cond_sigma,
            c=cond_pack["c"],
            cond_style=cond_pack.get("cond_style"),
            style_tokens=cond_pack.get("style_tokens"),
            style_attention_mask=cond_pack.get("style_attention_mask"),
            rotary_cos_sin=rotary_cos_sin,
            x_attention_mask=x_attention_mask,
            seqlens=seqlens,
        )

        # forward blocks
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            for blk in self.blocks:
                x = blk(
                    x,
                    ctx.rotary_cos_sin,
                    ctx.c,
                    seqlens=ctx.seqlens,
                    attention_mask=ctx.x_attention_mask,
                    style_tokens=ctx.style_tokens,
                    style_attention_mask=ctx.style_attention_mask,
                )

            x = self.output_layer(x, ctx.c)

        # optional scale-by-sigma logic (unchanged)
        if self.scale_by_sigma:
            assert self.absorb, "Haven't configured this to work."
            esigm1_log = (
                torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1)
                .log()
                .to(x.dtype)[:, None, None]
            )
            x = x - esigm1_log - np.log(x.shape[-1] - 1)

        # scatter zero at input ids (unchanged)
        x = torch.scatter(
            x, dim=-1, index=x_input_ids[..., None], src=torch.zeros_like(x[..., :1])
        )
        return x
