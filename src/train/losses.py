from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoder import CaptionEncoder
from model.utils import ScoreFn


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
        self.beta_cycle = None
        self.tau = config.training.tau

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
        return loss_sedd, pooled_h

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

    def _compute_cycle_loss(
        self,
        text,
        text_mask,
        style_caption,
    ) -> torch.Tensor:
        """
        caption reconstruction loss
        """
        if self.caption_decoder is None:
            raise NotImplementedError(
                "caption_decoder is None; cannot compute cycle loss."
            )
        # caption_pred: [B, cap_len, vocab_size]
        caption_pred = self.caption_decoder(text, text_mask)
        B, cap_len, V = caption_pred.shape
        caption_pred = caption_pred.view(B * cap_len, V)
        caption_gt = style_caption.view(B * cap_len)
        loss = F.cross_entropy(caption_pred, caption_gt, ignore_index=0)
        return loss

    def combine_losses(
        self,
        loss_sedd: torch.Tensor,
        loss_align: Optional[torch.Tensor] = None,
        loss_cycle: Optional[torch.Tensor] = None,
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

        loss_sedd = self._compute_sedd_loss(
            model=model,
            text=text,
            text_mask=text_mask,
            style_caption=style_caption,
            style_caption_mask=style_caption_mask,
            t=t,
            perturbed_batch=perturbed_batch,
        )

        return self.combine_losses(
            loss_sedd=loss_sedd,
            loss_align=None,
            loss_cycle=None,
        )


class SEDDLoss(LossFnBase):
    """
    total = loss_sedd
    """

    def combine_losses(self, loss_sedd, loss_align=None, loss_cycle=None):
        return loss_sedd


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

        loss_align = self._compute_align_loss(
            pooled_h=pooled_h,
            style_caption=style_caption,
            style_caption_mask=style_caption_mask,
            drop_indices=drop_indices,
        )

        return self.combine_losses(loss_sedd=loss_sedd, loss_align=loss_align)

    def combine_losses(self, loss_sedd, loss_align=None, loss_cycle=None):
        if loss_align is None:
            raise ValueError("loss_align is required for SEDDInfoNCELoss")
        return loss_sedd + self.alpha_align * loss_align


class SEDDInfoNCECycleLoss(LossFnBase):
    """
    total = loss_sedd + alpha_align * loss_align + beta_cycle * loss_cycle
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
        raise NotImplementedError(
            "SEDDInfoNCECycleLoss call not be used until caption_decoder is implemented."
        )
        style_caption, style_caption_mask = self._apply_cfg_dropout(
            style_caption, style_caption_mask, device=text.device
        )

        loss_sedd = self._compute_sedd_loss(
            model=model,
            text=text,
            text_mask=text_mask,
            style_caption=style_caption,
            style_caption_mask=style_caption_mask,
            t=t,
            perturbed_batch=perturbed_batch,
        )

        loss_align = self._compute_align_loss(
            text=text,
            text_mask=text_mask,
            style_caption=style_caption,
            style_caption_mask=style_caption_mask,
        )

        loss_cycle = self._compute_cycle_loss(
            text=text,
            text_mask=text_mask,
            style_caption=style_caption,
        )

        return self.combine_losses(
            loss_sedd=loss_sedd, loss_align=loss_align, loss_cycle=loss_cycle
        )

    def combine_losses(self, loss_sedd, loss_align=None, loss_cycle=None):
        if loss_align is None or loss_cycle is None:
            raise ValueError(
                "loss_align and loss_cycle are required for SEDDInfoNCECycleLoss"
            )
        return loss_sedd + self.alpha_align * loss_align + self.beta_cycle * loss_cycle


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
