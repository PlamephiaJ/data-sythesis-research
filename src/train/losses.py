from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from model.utils import ScoreFn


class LossFn:

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
        self.noise = noise
        self.graph = graph
        self.train = train
        self.sampling_eps = sampling_eps
        self.lv = lv
        self.p_uncond = p_uncond

        # ===== alignment & cycle hyperparameters =====
        self.alpha_align = config.alpha_align  # e.g., 0.3
        self.beta_cycle = config.beta_cycle  # e.g., 0.2
        self.tau = config.temperature  # InfoNCE temperature, e.g., 0.07

        # ===== encoders / decoder =====
        # caption_encoder and email_encoder produce normalized embeddings
        self.caption_encoder = config.caption_encoder
        self.email_encoder = config.email_encoder

        # caption_decoder: takes email tokens and predicts caption tokens
        self.caption_decoder = config.caption_decoder

    def _compute_info_nce_loss(self, emb_email, emb_caption):
        """
        emb_email: [B, D]
        emb_caption: [B, D]
        returns [B] InfoNCE loss per example
        """
        # normalize if not already normalized
        emb_email = F.normalize(emb_email, dim=-1)
        emb_caption = F.normalize(emb_caption, dim=-1)

        # compute similarity matrix [B, B]
        sim_matrix = emb_email @ emb_caption.T  # cosine logits
        sim_matrix = sim_matrix / self.tau

        # positive pairs sit on diagonal
        batch_size = sim_matrix.size(0)
        labels = torch.arange(batch_size, device=sim_matrix.device)

        # cross entropy loss on softmax of similarities
        loss = F.cross_entropy(sim_matrix, labels)
        return loss

    def _compute_cycle_loss(self, text, text_mask, style_caption):
        """
        caption reconstruction loss
        """
        # caption_pred: [B, cap_len, vocab_size]
        caption_pred = self.caption_decoder(text, text_mask)
        # flatten for CE
        B, cap_len, V = caption_pred.shape
        caption_pred = caption_pred.view(B * cap_len, V)
        caption_gt = style_caption.view(B * cap_len)
        loss = F.cross_entropy(caption_pred, caption_gt, ignore_index=0)
        return loss

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

        if t is None:
            if self.lv:
                raise NotImplementedError("LV branch not done")
            else:
                t = (1 - self.sampling_eps) * torch.rand(
                    text.shape[0], device=text.device
                ) + self.sampling_eps

        sigma, dsigma = self.noise(t)

        if perturbed_batch is None:
            perturbed_batch = self.graph.sample_transition(
                text, text_mask, sigma[:, None]
            )

        # classifier-free guidance dropout
        if self.train:
            b = text.shape[0]
            drop_indices = torch.rand(b, device=text.device) < self.p_uncond
            if drop_indices.any():
                style_caption_mask = style_caption_mask.clone()
                style_caption_mask[drop_indices] = 0
                style_caption = style_caption.clone()
                style_caption[drop_indices] = 0

        # ==== SEDD main loss ====
        log_score_fn = ScoreFn(model, train=self.train, sampling=False)
        log_score = log_score_fn(
            perturbed_batch, text_mask, style_caption, style_caption_mask, sigma
        )
        loss_sedd = self.graph.score_entropy(
            log_score, sigma[:, None], perturbed_batch, text, text_mask
        )
        loss_sedd = (dsigma[:, None] * loss_sedd).sum(dim=-1)

        # ==== alignment loss ====
        # get emb vectors
        emb_caption = self.caption_encoder(style_caption, style_caption_mask)
        emb_email = self.email_encoder(text, text_mask)

        # InfoNCE alignment
        loss_align = self._compute_info_nce_loss(emb_email, emb_caption)

        # ==== cycle consistency loss ====
        loss_cycle = self._compute_cycle_loss(text, text_mask, style_caption)

        # ==== total loss ====
        total_loss = (
            loss_sedd + self.alpha_align * loss_align + self.beta_cycle * loss_cycle
        )

        return total_loss


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
        self.train = train
        self.optimize_fn = optimize_fn
        self.accum = accum

        self.accum_iter = 0
        self.total_loss = 0

    def __call__(
        self, state, text, text_mask, style_caption, style_caption_mask, cond=None
    ):
        model = state["model"]

        if self.train:
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
