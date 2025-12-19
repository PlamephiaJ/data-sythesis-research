import abc

import torch

from model import utils as mutils
from sample.catsample import sample_categorical


_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(f"Already registered model with name: {local_name}")
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_predictor(name):
    return _PREDICTORS[name]


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, graph, noise):
        super().__init__()
        self.graph = graph
        self.noise = noise

    @abc.abstractmethod
    def update_fn(
        self, score_fn, x, x_mask, style_caption, style_caption_mask, t, step_size
    ):
        """One update of the predictor.

        Args:
            score_fn: score function
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.

        Returns:
            x: A PyTorch tensor of the next state.
        """
        pass


@register_predictor(name="euler")
class EulerPredictor(Predictor):

    def update_fn(
        self, score_fn, x, x_mask, style_caption, style_caption_mask, t, step_size
    ):
        sigma, dsigma = self.noise(t)
        score = score_fn(x, x_mask, style_caption, style_caption_mask, sigma)

        rev_rate = step_size * dsigma[..., None] * self.graph.reverse_rate(x, score)
        x = self.graph.sample_rate(x, rev_rate)
        return x


@register_predictor(name="none")
class NonePredictor(Predictor):

    def update_fn(self, score_fn, x, t, step_size):
        return x


@register_predictor(name="analytic")
class AnalyticPredictor(Predictor):

    def update_fn(
        self, score_fn, x, x_mask, style_caption, style_caption_mask, t, step_size
    ):
        curr_sigma = self.noise(t)[0]
        next_sigma = self.noise(t - step_size)[0]
        dsigma = curr_sigma - next_sigma

        score = score_fn(x, x_mask, style_caption, style_caption_mask, curr_sigma)

        # staggered score = 把在噪声 σ 下的 score，转换成在噪声 σ−dσ 下的 score 的近似解析形式
        stag_score = self.graph.staggered_score(score, dsigma)
        probs = stag_score * self.graph.transp_transition(x, dsigma)
        return sample_categorical(probs)


class Denoiser:

    def __init__(self, graph, noise):
        self.graph = graph
        self.noise = noise

    def update_fn(self, score_fn, x, x_mask, style_caption, style_caption_mask, t):
        sigma = self.noise(t)[0]

        score = score_fn(x, x_mask, style_caption, style_caption_mask, sigma)
        stag_score = self.graph.staggered_score(score, sigma)
        probs = stag_score * self.graph.transp_transition(x, sigma)
        # truncate probabilities
        if self.graph.absorb:
            probs = probs[..., :-1]

        # return probs.argmax(dim=-1)
        return sample_categorical(probs)


# def get_sampling_fn(config, graph, noise, batch_dims, eps, device):
#     sampling_fn = get_pc_sampler(
#         graph=graph,
#         noise=noise,
#         batch_dims=batch_dims,
#         predictor=config.sampling.predictor,
#         steps=config.sampling.steps,
#         denoise=config.sampling.noise_removal,
#         eps=eps,
#         device=device,
#     )

#     return sampling_fn


# def get_pc_sampler(
#     graph,
#     noise,
#     batch_dims,
#     predictor,
#     steps,
#     denoise=True,
#     eps=1e-5,
#     device=torch.device("cpu"),
#     proj_fun=lambda x: x,
# ):
#     predictor = get_predictor(predictor)(graph, noise)
#     projector = proj_fun
#     denoiser = Denoiser(graph, noise)

#     @torch.no_grad()
#     def pc_sampler(model):
#         sampling_score_fn = mutils.ScoreFn(model, train=False, sampling=True)
#         x = graph.sample_limit(*batch_dims).to(device)
#         # 从 1 到 eps 线性取 steps+1 个点，递减时间
#         timesteps = torch.linspace(1, eps, steps + 1, device=device)
#         dt = (1 - eps) / steps

#         for i in range(steps):
#             t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
#             x = projector(x)
#             x = predictor.update_fn(sampling_score_fn, x, t, dt)

#         if denoise:
#             # denoising step
#             x = projector(x)
#             t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
#             x = denoiser.update_fn(sampling_score_fn, x, t)

#         return x

#     return pc_sampler


class PCSampler:

    def __init__(
        self,
        graph,
        noise,
        batch_dims,
        predictor,
        steps,
        denoise=True,
        eps=1e-5,
        device=torch.device("cpu"),
        proj_fun=lambda x: x,
    ):
        self.graph = graph
        self.noise = noise
        self.batch_dims = batch_dims
        self.steps = steps
        self.denoise = denoise
        self.eps = eps
        self.device = device

        self.projector = proj_fun
        self.predictor = get_predictor(predictor)(graph, noise)
        self.denoiser = Denoiser(graph, noise)

        # 时间离散
        self.timesteps = torch.linspace(1, eps, steps + 1, device=device)
        self.dt = (1 - eps) / steps

    @torch.no_grad()
    def __call__(self, model, x_mask, style_caption, style_caption_mask):
        sampling_score_fn = mutils.ScoreFn(model, train=False, sampling=True)

        x = self.graph.sample_limit(*self.batch_dims).to(self.device)

        x_mask = x_mask[:, : self.batch_dims[1]].to(self.device)
        for i in range(self.steps):
            t = self.timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)

            x = self.projector(x)
            x = self.predictor.update_fn(
                sampling_score_fn,
                x,
                x_mask,
                style_caption,
                style_caption_mask,
                t,
                self.dt,
            )

        if self.denoise:
            x = self.projector(x)
            t = self.timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
            x = self.denoiser.update_fn(
                sampling_score_fn, x, x_mask, style_caption, style_caption_mask, t
            )

        print("sample output dtype/shape:", x.dtype, tuple(x.shape))
        return x
