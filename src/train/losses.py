import logging
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoder import CaptionEncoder
from model.utils import ScoreFn
from utils.tokenizer_factory import get_text_tokenizer


logger = logging.getLogger(__name__)


def _get_eos_id(config) -> int:
    if "tokenizer" not in config or "text" not in config.tokenizer:
        raise ValueError("Missing required config key: tokenizer.text")
    tokenizer_name = config.tokenizer.text
    tokenizer = get_text_tokenizer(tokenizer_name)
    return int(tokenizer.eos_token_id)


def _sample_t(batch_size: int, device: torch.device, sampling_eps: float, lv: bool):
    if lv:
        raise NotImplementedError("LV branch not done")
    return (1 - sampling_eps) * torch.rand(batch_size, device=device) + sampling_eps


def _apply_cfg_dropout(
    style_caption: Optional[torch.Tensor],
    style_caption_mask: Optional[torch.Tensor],
    *,
    is_train: bool,
    p_uncond: float,
    device: torch.device,
):
    if style_caption is None or style_caption_mask is None:
        drop_indices = torch.zeros(0, device=device, dtype=torch.bool)
        return style_caption, style_caption_mask, drop_indices

    if not is_train:
        b = style_caption.shape[0]
        drop_indices = torch.zeros(b, device=device, dtype=torch.bool)
        return style_caption, style_caption_mask, drop_indices

    b = style_caption.shape[0]
    drop_indices = torch.rand(b, device=device) < p_uncond
    if drop_indices.any():
        style_caption_mask = style_caption_mask.clone()
        style_caption_mask[drop_indices] = 0
        style_caption = style_caption.clone()
        style_caption[drop_indices] = 0
    return style_caption, style_caption_mask, drop_indices


def _eos_penalty_term(log_score, text, text_mask, eos_id: int):
    eos_mask = (text == eos_id) & text_mask.bool()
    if not eos_mask.any():
        return log_score.new_zeros((text.shape[0],))
    log_probs = F.log_softmax(log_score, dim=-1)
    eos_logp = log_probs[..., eos_id]
    eos_loss = -(eos_logp * eos_mask).sum(dim=-1)
    denom = eos_mask.sum(dim=-1).clamp(min=1)
    return eos_loss / denom


def _compute_info_nce_loss(
    emb_email: torch.Tensor, emb_caption: torch.Tensor, tau: float
):
    emb_email = F.normalize(emb_email, dim=-1)
    emb_caption = F.normalize(emb_caption, dim=-1)
    sim = (emb_email @ emb_caption.T) / tau
    labels = torch.arange(sim.size(0), device=sim.device)
    loss_e2c = F.cross_entropy(sim, labels, reduction="mean")
    loss_c2e = F.cross_entropy(sim.T, labels, reduction="mean")
    return 0.5 * (loss_e2c + loss_c2e)


def _normalize_loss_terms(config) -> List[Dict[str, float]]:
    if "training" not in config or "loss_terms" not in config.training:
        raise ValueError("`training.loss_terms` is required in config")
    terms_cfg = config.training.loss_terms
    if not terms_cfg:
        raise ValueError("`training.loss_terms` must not be empty")

    terms = []
    for item in terms_cfg:
        if isinstance(item, str):
            raise ValueError(
                "String loss term is not allowed. Use mapping: {name: <term>, weight: <float>}"
            )

        name = item.get("name")
        if not name:
            raise ValueError("Each entry in `training.loss_terms` must include `name`")
        if "weight" not in item:
            raise ValueError(
                f"Loss term '{name}' must include explicit `weight` in `training.loss_terms`"
            )
        terms.append(
            {
                "name": name,
                "weight": float(item["weight"]),
                "eos_id": item.get("eos_id"),
            }
        )

    return terms


def build_loss_fn(
    config,
    noise: Callable,
    graph,
    train: bool,
    *,
    sampling_eps: float = 1e-3,
    lv: bool = False,
    p_uncond: float = 0.1,
    return_details: bool = False,
):
    terms = [t for t in _normalize_loss_terms(config) if float(t["weight"]) > 0]
    allowed = {"sedd", "align", "eos_penalty"}
    for t in terms:
        if t["name"] not in allowed:
            raise ValueError(
                f"Unknown loss term: {t['name']}. Allowed: {sorted(allowed)}"
            )

    term_weights = {t["name"]: float(t["weight"]) for t in terms}
    logger.info(
        "Enabled loss terms (%s): %s",
        "train" if train else "eval",
        ", ".join(f"{name}={coef:.6g}" for name, coef in term_weights.items()),
    )
    use_align = "align" in term_weights and term_weights["align"] > 0
    use_eos = "eos_penalty" in term_weights and term_weights["eos_penalty"] > 0

    if use_align:
        if "data" not in config or "trainset" not in config.data:
            raise ValueError("Missing required config key: data.trainset")
        if "validset" not in config.data:
            raise ValueError("Missing required config key: data.validset")
        if "format" not in config.data.trainset:
            raise ValueError("Missing required config key: data.trainset.format")
        if "format" not in config.data.validset:
            raise ValueError("Missing required config key: data.validset.format")

        train_format = str(config.data.trainset.format).strip().lower()
        valid_format = str(config.data.validset.format).strip().lower()
        if train_format != "entry" or valid_format != "entry":
            raise ValueError(
                "align loss requires entry-format datasets for both trainset and validset."
            )

    eos_id = next(
        (
            int(t.get("eos_id"))
            for t in terms
            if t["name"] == "eos_penalty" and t.get("eos_id") is not None
        ),
        _get_eos_id(config),
    )

    caption_encoder = None
    email_proj = None
    if "tau" not in config.training:
        raise ValueError("Missing required config key: training.tau")
    tau = float(config.training.tau)
    if use_align:
        if "caption_encoder" not in config.model:
            raise ValueError("align loss needs `model.caption_encoder` in config")
        caption_encoder = CaptionEncoder(
            name=config.model.caption_encoder.name,
            cond_dim=config.model.cond_dim,
            pool=config.model.caption_encoder.pool,
            dropout=config.model.caption_encoder.dropout,
            freeze=config.model.caption_encoder.freeze,
            token_dim=config.model.hidden_size,
            device=None,
        )
        email_proj = nn.Linear(
            config.model.hidden_size, config.model.cond_dim, bias=False
        )
        nn.init.normal_(email_proj.weight, std=0.02)

    log_score_fn = ScoreFn(None, train=train, sampling=False)

    def loss_fn(
        model,
        text,
        text_mask,
        style_caption,
        style_caption_mask,
        cond=None,
        t: Optional[torch.Tensor] = None,
        perturbed_batch: Optional[torch.Tensor] = None,
    ):
        del cond
        nonlocal caption_encoder, email_proj

        component_losses = {}

        style_caption, style_caption_mask, drop_indices = _apply_cfg_dropout(
            style_caption,
            style_caption_mask,
            is_train=train,
            p_uncond=p_uncond,
            device=text.device,
        )

        if t is None:
            t = _sample_t(text.shape[0], text.device, sampling_eps=sampling_eps, lv=lv)
        sigma, dsigma = noise(t)

        if perturbed_batch is None:
            perturbed_batch = graph.sample_transition(text, text_mask, sigma[:, None])

        # Rebind wrapped model each step (keeps closure simple)
        log_score_fn.model_fn.model = model

        pooled_h = None
        if use_align:
            if style_caption is None or style_caption_mask is None:
                raise ValueError(
                    "align loss is enabled but style_caption inputs are missing"
                )

            if (
                caption_encoder is not None
                and next(caption_encoder.parameters()).device != text.device
            ):
                caption_encoder = caption_encoder.to(text.device)
            if email_proj is not None and email_proj.weight.device != text.device:
                email_proj = email_proj.to(text.device)

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

        loss_sedd = graph.score_entropy(
            log_score, sigma[:, None], perturbed_batch, text, text_mask
        )
        loss_sedd = (dsigma[:, None] * loss_sedd).sum(dim=-1)
        component_losses["sedd"] = loss_sedd.mean().detach()

        total = log_score.new_zeros(loss_sedd.shape)
        if "sedd" in term_weights and term_weights["sedd"] > 0:
            total = total + term_weights["sedd"] * loss_sedd

        if use_align:
            keep = (~drop_indices) & (style_caption_mask.sum(dim=1) > 0)
            if keep.sum().item() >= 2:
                cap_k = style_caption[keep]
                cap_mask_k = style_caption_mask[keep]
                pooled_k = pooled_h[keep].float()

                with torch.no_grad():
                    emb_caption = caption_encoder(
                        input_ids=cap_k,
                        attention_mask=cap_mask_k,
                        return_align=True,
                    )

                emb_email = F.normalize(email_proj(pooled_k), dim=-1)
                loss_align = _compute_info_nce_loss(emb_email, emb_caption, tau=tau)
                total = total + term_weights["align"] * loss_align
                component_losses["infonce"] = loss_align.detach()
            else:
                component_losses["infonce"] = total.new_zeros((), device=total.device)
        elif "align" in term_weights and term_weights["align"] > 0:
            component_losses["infonce"] = total.new_zeros((), device=total.device)

        if use_eos:
            eos_term = _eos_penalty_term(log_score, text, text_mask, eos_id=eos_id)
            total = total + term_weights["eos_penalty"] * eos_term
            component_losses["eos_prediction"] = eos_term.mean().detach()
        elif "eos_penalty" in term_weights and term_weights["eos_penalty"] > 0:
            component_losses["eos_prediction"] = total.new_zeros(
                (), device=total.device
            )

        if return_details:
            return total, component_losses
        return total

    return loss_fn


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
        self.total_details = None

    def __call__(
        self, state, text, text_mask, style_caption, style_caption_mask, cond=None
    ):
        model = state["model"]

        if self.is_train:
            optimizer = state["optimizer"]
            scaler = state["scaler"]

            loss_out = self.loss_fn(
                model, text, text_mask, style_caption, style_caption_mask, cond=cond
            )
            if isinstance(loss_out, tuple):
                loss_tensor, details = loss_out
            else:
                loss_tensor, details = loss_out, None

            loss = loss_tensor.mean() / self.accum
            scaler.scale(loss).backward()

            self.accum_iter += 1
            self.total_loss += loss.detach()
            if details is not None:
                if self.total_details is None:
                    self.total_details = {
                        key: value.detach() / self.accum
                        for key, value in details.items()
                    }
                else:
                    for key, value in details.items():
                        if key not in self.total_details:
                            self.total_details[key] = value.detach() / self.accum
                        else:
                            self.total_details[key] += value.detach() / self.accum

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
                details_to_return = self.total_details
                self.total_loss = 0
                self.total_details = None
                if details_to_return is not None:
                    return loss_to_return, details_to_return
                return loss_to_return

            # micro-step: return current scaled micro loss (same semantics as your original)
            if details is not None:
                return loss, {
                    key: value.detach() / self.accum for key, value in details.items()
                }
            return loss

        # eval
        with torch.no_grad():
            ema = state["ema"]
            ema.store(model.parameters())
            ema.copy_to(model.parameters())

            loss = self.loss_fn(
                model, text, text_mask, style_caption, style_caption_mask, cond=cond
            )

            if isinstance(loss, tuple):
                loss_tensor, details = loss
                loss = loss_tensor.mean()
            else:
                details = None
                loss = loss.mean()

            ema.restore(model.parameters())
            if details is not None:
                return loss, {key: value.detach() for key, value in details.items()}
            return loss
