# def get_loss_fn(noise, graph, train, sampling_eps=1e-3, lv=False):
#     def loss_fn(model, text, style_caption, cond=None, t=None, perturbed_batch=None):
#         """
#         Batch shape: [B, L] int. D given from graph
#         """
#         if t is None:
#             if lv:
#                 raise NotImplementedError("Yeah I gotta do this later")
#             else:
#                 # t 是连续时间维度上的采样，均匀分布采样，范围是 [sampling_eps, 1]
#                 # 加入 sampling_eps 是为了避免 sigma=0 的情况
#                 # t shape: [B]
#                 t = (1 - sampling_eps) * torch.rand(
#                     text.shape[0], device=text.device
#                 ) + sampling_eps
#         # sigma: [B], dsigma: [B]
#         sigma, dsigma = noise(t)
#         if perturbed_batch is None:
#             perturbed_batch = graph.sample_transition(text, sigma[:, None])
#         log_score_fn = mutils.get_score_fn(model, train=train, sampling=False)
#         # log_score: [B, seqlen, vocab_size], 即每个位置上每个token的log得分
#         # 在score model输出最终将跳转到自己的score设置为了0，其他地方是模型预测的log score
#         log_score = log_score_fn(perturbed_batch, style_caption, sigma)
#         loss = graph.score_entropy(log_score, sigma[:, None], perturbed_batch, text)
#         loss = (dsigma[:, None] * loss).sum(dim=-1)
#         return loss
#     return loss_fn
from typing import Callable, Optional

import numpy as np
import torch

from model import utils as mutils


class LossFn:

    def __init__(
        self,
        noise: Callable,
        graph,
        train: bool,
        sampling_eps: float = 1e-3,
        lv: bool = False,
    ):
        """
        Args:
            noise: callable, input t -> (sigma, dsigma), both shape [B]
            graph: object providing sample_transition(...) and score_entropy(...)
            train: bool, passed to get_score_fn
            sampling_eps: float, avoid sigma=0
            lv: bool, whether to use lv branch
        """
        self.noise = noise
        self.graph = graph
        self.train = train
        self.sampling_eps = sampling_eps
        self.lv = lv

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
        """
        Batch shape: text [B, L] int. D given from graph
        """
        if t is None:
            if self.lv:
                raise NotImplementedError("Yeah I gotta do this later")
            else:
                # t uniform in [sampling_eps, 1], shape [B]
                t = (1 - self.sampling_eps) * torch.rand(
                    text.shape[0], device=text.device
                ) + self.sampling_eps

        # sigma: [B], dsigma: [B]
        sigma, dsigma = self.noise(t)

        if perturbed_batch is None:
            perturbed_batch = self.graph.sample_transition(
                text, text_mask, sigma[:, None]
            )

        log_score_fn = mutils.get_score_fn(model, train=self.train, sampling=False)
        log_score = log_score_fn(
            perturbed_batch, style_caption, style_caption_mask, sigma
        )

        loss = self.graph.score_entropy(
            log_score, sigma[:, None], perturbed_batch, text
        )

        loss = (dsigma[:, None] * loss).sum(dim=-1)
        return loss


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

            loss = self.loss_fn(model, text, style_caption, cond=cond).mean()

            ema.restore(model.parameters())
            return loss
