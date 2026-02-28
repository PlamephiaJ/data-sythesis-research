import torch


# def get_model_fn(model, train=False):
#     """Create a function to give the output of the score-based model.

#     Args:
#         model: The score model.
#         train: `True` for training and `False` for evaluation.
#         mlm: If the input model is a mlm and models the base probability

#     Returns:
#         A model function.
#     """

#     def model_fn(x, style_caption, sigma):
#         """Compute the output of the score-based model.

#         Args:
#             x: A mini-batch of input data.
#             labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
#               for different models.

#         Returns:
#             A tuple of (model output, new mutable states)
#         """
#         if train:
#             model.train()
#         else:
#             model.eval()

#             # otherwise output the raw values (we handle mlm training in losses.py)
#         return model(x, style_caption, sigma)

#     return model_fn


# def get_score_fn(model, train=False, sampling=False):
#     if sampling:
#         assert not train, "Must sample in eval mode"
#     model_fn = get_model_fn(model, train=train)

#     def score_fn(x, style_caption, sigma):
#         device = x.device
#         with torch.amp.autocast(
#             device_type=device.type,
#             dtype=torch.bfloat16,
#             enabled=(device.type == "cuda"),
#         ):
#             sigma = sigma.reshape(-1)
#             score = model_fn(x, style_caption, sigma)

#         if sampling:
#             # when sampling return true score (not log used for training)
#             return score.exp()

#         return score

#     return score_fn


class ModelFn:
    """Thin wrapper: set train/eval then call model."""

    def __init__(self, model, train: bool = False):
        self.model = model
        self.train = train

    def __call__(self, x, x_mask, style_caption, style_caption_mask, sigma, **kwargs):
        if self.train:
            self.model.train()
        else:
            self.model.eval()
        return self.model(x, x_mask, style_caption, style_caption_mask, sigma, **kwargs)


class ScoreFn:
    """Wrap ModelFn with autocast and optional exp() at sampling time."""

    def __init__(self, model, train: bool = False, sampling: bool = False):
        if sampling and train:
            raise ValueError("Must sample in eval mode")
        self.sampling = sampling
        self.model_fn = ModelFn(model, train=train)

    def __call__(self, x, x_mask, style_caption, style_caption_mask, sigma, **kwargs):
        # Optional CFG parameters (only used during sampling)
        cfg_scale = kwargs.pop("cfg_scale", 0.0)
        use_cfg = (
            self.sampling
            and (cfg_scale is not None)
            and (cfg_scale > 0)
            and (style_caption is not None)
            and (style_caption_mask is not None)
        )

        device = x.device
        sigma = sigma.reshape(-1)

        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda"),
        ):
            # Conditional prediction (always computed)
            out_cond = self.model_fn(
                x, x_mask, style_caption, style_caption_mask, sigma, **kwargs
            )

            if use_cfg:
                # Construct unconditional condition (consistent with your training dropout)
                style_caption_u = torch.zeros_like(style_caption)
                style_caption_mask_u = torch.zeros_like(style_caption_mask)

                out_uncond = self.model_fn(
                    x, x_mask, style_caption_u, style_caption_mask_u, sigma, **kwargs
                )

                # Combine in log-space (before exp)
                out = (1.0 + cfg_scale) * out_cond - cfg_scale * out_uncond
            else:
                out = out_cond

        # Preserve your existing tuple-handling behavior
        if isinstance(out, tuple):
            score, *rest = out
            score = score.exp() if self.sampling else score
            return (score, *rest)
        else:
            return out.exp() if self.sampling else out
