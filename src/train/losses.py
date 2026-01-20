from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoder import CaptionEncoder
from model.utils import ScoreFn
from utils.tokenizer_factory import get_text_tokenizer


@dataclass
class AuxLoss:
    name: str
    weight: float
    requires: Sequence[str]

    def __call__(self, **ctx) -> torch.Tensor:
        raise NotImplementedError


class EosPenaltyLoss(AuxLoss):

    def __init__(self, weight: float, eos_id: int):
        super().__init__(
            name="eos_penalty",
            weight=weight,
            requires=("log_score", "text", "text_mask"),
        )
        self.eos_id = int(eos_id)

    def __call__(self, **ctx) -> torch.Tensor:
        log_score = ctx["log_score"]
        text = ctx["text"]
        text_mask = ctx["text_mask"]

        eos_mask = (text == self.eos_id) & text_mask.bool()
        if not eos_mask.any():
            return log_score.new_zeros((text.shape[0],))
        log_probs = F.log_softmax(log_score, dim=-1)
        eos_logp = log_probs[..., self.eos_id]
        eos_loss = -(eos_logp * eos_mask).sum(dim=-1)
        denom = eos_mask.sum(dim=-1).clamp(min=1)
        return eos_loss / denom


def _get_eos_id(config) -> int:
    tokenizer_name = getattr(config.tokenizer, "text", "gpt2")
    tokenizer = get_text_tokenizer(tokenizer_name)
    return int(tokenizer.eos_token_id)


def _build_aux_losses(config) -> List[AuxLoss]:
    losses: List[AuxLoss] = []

    aux_cfg = getattr(config.training, "aux_losses", None)
    if aux_cfg:
        for item in aux_cfg:
            name = item.get("name")
            weight = float(item.get("weight", 0.0))
            if weight <= 0:
                continue
            if name == "eos_penalty":
                eos_id = item.get("eos_id", _get_eos_id(config))
                losses.append(EosPenaltyLoss(weight=weight, eos_id=eos_id))
            else:
                raise ValueError(f"Unknown auxiliary loss: {name}")
    else:
        eos_penalty = float(getattr(config.training, "eos_penalty", 0.0))
        if eos_penalty > 0:
            losses.append(
                EosPenaltyLoss(weight=eos_penalty, eos_id=_get_eos_id(config))
            )

    return losses


class LossFnBase(nn.Module):

    def __init__(
        self,
        config,
        noise: Callable,
        graph,
        train: bool,
        sampling_eps: float = 1e-3,
        lv: bool = False,
        p_uncond: float = 0.1,
    ):
        super().__init__()
        self.noise = noise
        self.graph = graph
        self.is_train = train
        self.sampling_eps = sampling_eps
        self.lv = lv
        self.p_uncond = p_uncond

        # ===== hyperparameters =====
        self.alpha_align = config.training.alpha_align
        self.tau = config.training.tau
        self.aux_losses = _build_aux_losses(config)
        self._aux_requires_log_score = any(
            "log_score" in loss.requires for loss in self.aux_losses
        )

        # ===== encoders / decoder =====
        self.caption_encoder = CaptionEncoder(
            name=config.model.caption_encoder.name,
            cond_dim=config.model.cond_dim,
            pool=config.model.caption_encoder.pool,
            dropout=config.model.caption_encoder.dropout,
            freeze=config.model.caption_encoder.freeze,
            token_dim=config.model.hidden_size,
            device=next(self.parameters()).device,
        )

        self.email_proj = nn.Linear(
            config.model.hidden_size, config.model.cond_dim, bias=False
        ).to(device=next(self.parameters()).device)
        nn.init.normal_(self.email_proj.weight, std=0.02)

        self.caption_decoder = None

    # -------------------------
    # helpers
    # -------------------------
    def _sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.lv:
            raise NotImplementedError("LV branch not done")
        t = (1 - self.sampling_eps) * torch.rand(
            batch_size, device=device
        ) + self.sampling_eps
        return t

    def _apply_cfg_dropout(
        self,
        style_caption: torch.Tensor,
        style_caption_mask: torch.Tensor,
        device: torch.device,
    ):
        if not self.is_train:
            b = style_caption.shape[0]
            drop_indices = torch.zeros(b, device=device, dtype=torch.bool)
            return style_caption, style_caption_mask, drop_indices

        b = style_caption.shape[0]
        drop_indices = torch.rand(b, device=device) < self.p_uncond
        if drop_indices.any():
            style_caption_mask = style_caption_mask.clone()
            style_caption_mask[drop_indices] = 0
            style_caption = style_caption.clone()
            style_caption[drop_indices] = 0
        return style_caption, style_caption_mask, drop_indices

    def _compute_sedd_loss(
        self,
        model,
        text,
        text_mask,
        style_caption,
        style_caption_mask,
        t: Optional[torch.Tensor] = None,
        perturbed_batch: Optional[torch.Tensor] = None,
        return_log_score: bool = False,
    ) -> torch.Tensor:
        if t is None:
            t = self._sample_t(batch_size=text.shape[0], device=text.device)

        sigma, dsigma = self.noise(t)

        if perturbed_batch is None:
            perturbed_batch = self.graph.sample_transition(
                text, text_mask, sigma[:, None]
            )

        log_score_fn = ScoreFn(model, train=self.is_train, sampling=False)
        log_score = log_score_fn(
            perturbed_batch, text_mask, style_caption, style_caption_mask, sigma
        )

        loss_sedd = self.graph.score_entropy(
            log_score, sigma[:, None], perturbed_batch, text, text_mask
        )
        loss_sedd = (dsigma[:, None] * loss_sedd).sum(dim=-1)  # [B]
        if return_log_score:
            return loss_sedd, log_score
        return loss_sedd

    def _compute_sedd_loss_and_pooled(
        self,
        model,
        text,
        text_mask,
        style_caption,
        style_caption_mask,
        t=None,
        perturbed_batch=None,
        return_pooled: bool = False,
        return_log_score: bool = False,
    ):
        if t is None:
            t = self._sample_t(batch_size=text.shape[0], device=text.device)
        sigma, dsigma = self.noise(t)

        if perturbed_batch is None:
            perturbed_batch = self.graph.sample_transition(
                text, text_mask, sigma[:, None]
            )

        log_score_fn = ScoreFn(model, train=self.is_train, sampling=False)

        if return_pooled:
            log_score, pooled_h = log_score_fn(
                perturbed_batch,
                text_mask,
                style_caption,
                style_caption_mask,
                sigma,
                return_pooled=True,
            )
        else:
            log_score = log_score_fn(
                perturbed_batch,
                text_mask,
                style_caption,
                style_caption_mask,
                sigma,
            )
            pooled_h = None

        loss_sedd = self.graph.score_entropy(
            log_score, sigma[:, None], perturbed_batch, text, text_mask
        )
        loss_sedd = (dsigma[:, None] * loss_sedd).sum(dim=-1)  # [B]
        if return_log_score:
            return loss_sedd, pooled_h, log_score
        return loss_sedd, pooled_h

    def _compute_aux_total(self, **ctx) -> Optional[torch.Tensor]:
        if not self.aux_losses:
            return None
        losses: List[torch.Tensor] = []
        for loss in self.aux_losses:
            losses.append(loss(**ctx) * loss.weight)
        return sum(losses)

    def _compute_info_nce_loss(
        self,
        emb_email: torch.Tensor,  # [B, D]
        emb_caption: torch.Tensor,  # [B, D]
    ) -> torch.Tensor:

        emb_email = F.normalize(emb_email, dim=-1)
        emb_caption = F.normalize(emb_caption, dim=-1)

        # similarity matrix
        sim = (emb_email @ emb_caption.T) / self.tau  # [B, B]

        labels = torch.arange(sim.size(0), device=sim.device)

        # email -> caption
        loss_e2c = F.cross_entropy(sim, labels, reduction="mean")

        # caption -> email
        loss_c2e = F.cross_entropy(sim.T, labels, reduction="mean")

        return 0.5 * (loss_e2c + loss_c2e)

    def _compute_align_loss(
        self,
        pooled_h: torch.Tensor,  # [B, H] from diffusion (same forward as sedd loss)
        style_caption: torch.Tensor,  # [B, Lc]
        style_caption_mask: torch.Tensor,  # [B, Lc]
        drop_indices: torch.Tensor,  # [B] True means caption dropped
    ) -> torch.Tensor:

        keep = (~drop_indices) & (style_caption_mask.sum(dim=1) > 0)

        # InfoNCE needs at least 2 samples in-batch
        if keep.sum().item() < 2:
            return pooled_h.new_zeros(())  # scalar 0 on correct device/dtype

        cap_k = style_caption[keep]
        cap_mask_k = style_caption_mask[keep]
        pooled_k = pooled_h[keep].float()  # [Bv, H]

        # caption target (frozen)
        with torch.no_grad():
            emb_caption = self.caption_encoder(
                input_ids=cap_k,
                attention_mask=cap_mask_k,
                return_align=True,
            )  # [Bv, D]

        # email embedding predicted by diffusion hidden (THIS is the key)
        emb_email = F.normalize(self.email_proj(pooled_k), dim=-1)  # [Bv, D]

        return self._compute_info_nce_loss(emb_email, emb_caption)

    def combine_losses(
        self,
        loss_sedd: torch.Tensor,
        loss_align: Optional[torch.Tensor] = None,
        aux_loss: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Rewrite in subclass to combine losses as needed.
        """
        raise NotImplementedError

    def __call__(
        self,
        model,
        text,
        text_mask,
        style_caption,
        style_caption_mask,
        cond=None,
        t: Optional[torch.Tensor] = None,
        perturbed_batch: Optional[torch.Tensor] = None,
    ):
        # CFG dropout (only affects conditions, not the text itself)
        style_caption, style_caption_mask, _ = self._apply_cfg_dropout(
            style_caption, style_caption_mask, device=text.device
        )

        if self._aux_requires_log_score:
            loss_sedd, log_score = self._compute_sedd_loss(
                model=model,
                text=text,
                text_mask=text_mask,
                style_caption=style_caption,
                style_caption_mask=style_caption_mask,
                t=t,
                perturbed_batch=perturbed_batch,
                return_log_score=True,
            )
            aux_loss = self._compute_aux_total(
                log_score=log_score,
                text=text,
                text_mask=text_mask,
                style_caption=style_caption,
                style_caption_mask=style_caption_mask,
            )
        else:
            loss_sedd = self._compute_sedd_loss(
                model=model,
                text=text,
                text_mask=text_mask,
                style_caption=style_caption,
                style_caption_mask=style_caption_mask,
                t=t,
                perturbed_batch=perturbed_batch,
            )
            aux_loss = self._compute_aux_total(
                text=text,
                text_mask=text_mask,
                style_caption=style_caption,
                style_caption_mask=style_caption_mask,
            )

        return self.combine_losses(
            loss_sedd=loss_sedd,
            loss_align=None,
            aux_loss=aux_loss,
        )


class SEDDLoss(LossFnBase):
    """
    total = loss_sedd
    """

    def combine_losses(self, loss_sedd, loss_align=None, aux_loss=None):
        if aux_loss is None:
            return loss_sedd
        return loss_sedd + aux_loss


class SEDDInfoNCELoss(LossFnBase):
    """
    total = loss_sedd + alpha_align * loss_align
    """

    def __call__(
        self,
        model,
        text,
        text_mask,
        style_caption,
        style_caption_mask,
        cond=None,
        t: Optional[torch.Tensor] = None,
        perturbed_batch: Optional[torch.Tensor] = None,
    ):
        style_caption, style_caption_mask, drop_indices = self._apply_cfg_dropout(
            style_caption, style_caption_mask, device=text.device
        )

        if self._aux_requires_log_score:
            loss_sedd, pooled_h, log_score = self._compute_sedd_loss_and_pooled(
                model=model,
                text=text,
                text_mask=text_mask,
                style_caption=style_caption,
                style_caption_mask=style_caption_mask,
                t=t,
                perturbed_batch=perturbed_batch,
                return_pooled=True,
                return_log_score=True,
            )
            aux_loss = self._compute_aux_total(
                log_score=log_score,
                text=text,
                text_mask=text_mask,
                style_caption=style_caption,
                style_caption_mask=style_caption_mask,
            )
        else:
            loss_sedd, pooled_h = self._compute_sedd_loss_and_pooled(
                model=model,
                text=text,
                text_mask=text_mask,
                style_caption=style_caption,
                style_caption_mask=style_caption_mask,
                t=t,
                perturbed_batch=perturbed_batch,
                return_pooled=True,
            )
            aux_loss = self._compute_aux_total(
                text=text,
                text_mask=text_mask,
                style_caption=style_caption,
                style_caption_mask=style_caption_mask,
            )

        loss_align = self._compute_align_loss(
            pooled_h=pooled_h,
            style_caption=style_caption,
            style_caption_mask=style_caption_mask,
            drop_indices=drop_indices,
        )

        return self.combine_losses(
            loss_sedd=loss_sedd, loss_align=loss_align, aux_loss=aux_loss
        )

    def combine_losses(self, loss_sedd, loss_align=None, aux_loss=None):
        if loss_align is None:
            raise ValueError("loss_align is required for SEDDInfoNCELoss")
        total = loss_sedd + self.alpha_align * loss_align
        if aux_loss is not None:
            total = total + aux_loss
        return total


# def optimization_manager(config):
#     """Returns an optimize_fn based on `config`."""

#     def optimize_fn(
#         optimizer,
#         scaler,
#         params,
#         step,
#         lr=config.optim.lr,
#         warmup=config.optim.warmup,
#         grad_clip=config.optim.grad_clip,
#     ):
#         """Optimizes with warmup and gradient clipping (disabled if negative)."""
#         scaler.unscale_(optimizer)

#         if warmup > 0:
#             for g in optimizer.param_groups:
#                 g["lr"] = lr * np.minimum(step / warmup, 1.0)
#         if grad_clip >= 0:
#             torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)

#         scaler.step(optimizer)
#         scaler.update()

#     return optimize_fn


class OptimizationManager:
    """Callable optimization step manager with warmup and grad clipping."""

    def __init__(self, config):
        self.config = config

    def __call__(
        self,
        optimizer,
        scaler,
        params,
        step,
        lr=None,
        warmup=None,
        grad_clip=None,
    ):
        """
        Optimizes with warmup and gradient clipping (disabled if negative).
        Args:
            optimizer: torch optimizer
            scaler: torch.cuda.amp.GradScaler
            params: iterable of parameters (for grad clipping)
            step: int/float, current global step
            lr/warmup/grad_clip: if None, use config.optim.*
        """
        # defaults from config
        if lr is None:
            lr = self.config.optim.lr
        if warmup is None:
            warmup = self.config.optim.warmup
        if grad_clip is None:
            grad_clip = self.config.optim.grad_clip

        # Unscale before clipping
        scaler.unscale_(optimizer)

        # LR warmup
        if warmup and warmup > 0:
            warmup_factor = np.minimum(step / warmup, 1.0)
            for g in optimizer.param_groups:
                g["lr"] = lr * warmup_factor

        # Grad clipping
        if grad_clip is not None and grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)

        # Optimizer step via scaler
        scaler.step(optimizer)
        scaler.update()


# def get_step_fn(noise, graph, train, optimize_fn, accum):
#     loss_fn = get_loss_fn(noise, graph, train)

#     accum_iter = 0
#     total_loss = 0

#     def step_fn(state, text, style_caption, cond=None):
#         nonlocal accum_iter
#         nonlocal total_loss

#         model = state["model"]

#         if train:
#             optimizer = state["optimizer"]
#             scaler = state["scaler"]
#             loss = loss_fn(model, text, style_caption, cond=cond).mean() / accum

#             scaler.scale(loss).backward()

#             accum_iter += 1
#             total_loss += loss.detach()
#             if accum_iter == accum:
#                 accum_iter = 0

#                 state["step"] += 1
#                 optimize_fn(optimizer, scaler, model.parameters(), step=state["step"])
#                 state["ema"].update(model.parameters())
#                 optimizer.zero_grad()

#                 loss = total_loss
#                 total_loss = 0
#         else:
#             with torch.no_grad():
#                 ema = state["ema"]
#                 ema.store(model.parameters())
#                 ema.copy_to(model.parameters())
#                 loss = loss_fn(model, text, style_caption, cond=cond).mean()
#                 ema.restore(model.parameters())

#         return loss

#     return step_fn


class StepFn:

    def __init__(self, *, loss_fn, train: bool, optimize_fn, accum: int):
        """
        Args:
            loss_fn: callable, signature like loss_fn(model, text, style_caption, cond=None, ...)
                    returns per-sample loss [B] or scalar; this StepFn will .mean()
            train: bool
            optimize_fn: callable, signature optimize_fn(optimizer, scaler, params, step, ...)
            accum: gradient accumulation steps
        """
        if accum <= 0:
            raise ValueError(f"accum must be positive, got {accum}")

        self.loss_fn = loss_fn
        self.is_train = train
        self.optimize_fn = optimize_fn
        self.accum = accum

        self.accum_iter = 0
        self.total_loss = 0

    def __call__(
        self, state, text, text_mask, style_caption, style_caption_mask, cond=None
    ):
        model = state["model"]

        if self.is_train:
            optimizer = state["optimizer"]
            scaler = state["scaler"]

            loss = (
                self.loss_fn(
                    model, text, text_mask, style_caption, style_caption_mask, cond=cond
                ).mean()
                / self.accum
            )
            scaler.scale(loss).backward()

            self.accum_iter += 1
            self.total_loss += loss.detach()

            if self.accum_iter == self.accum:
                self.accum_iter = 0

                state["step"] += 1
                self.optimize_fn(
                    optimizer=optimizer,
                    scaler=scaler,
                    params=model.parameters(),
                    step=state["step"],
                )
                state["ema"].update(model.parameters())
                optimizer.zero_grad(set_to_none=True)

                loss_to_return = self.total_loss
                self.total_loss = 0
                return loss_to_return

            # micro-step: return current scaled micro loss (same semantics as your original)
            return loss

        # eval
        with torch.no_grad():
            ema = state["ema"]
            ema.store(model.parameters())
            ema.copy_to(model.parameters())

            loss = self.loss_fn(
                model, text, text_mask, style_caption, style_caption_mask, cond=cond
            ).mean()

            ema.restore(model.parameters())
            return loss
