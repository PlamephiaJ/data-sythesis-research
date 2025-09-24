import os

import torch

from ..models import SEDD
from ..models.ema import ExponentialMovingAverage
from ..utils.dist import load_hydra_config_from_run
from ..utils.graph_noise import get_graph, get_noise


def load_model_hf(dir, device):
    score_model = SEDD.from_pretrained(dir).to(device)
    graph = get_graph(score_model.config, device)
    noise = get_noise(score_model.config).to(device)
    return score_model, graph, noise


def load_model_local(root_dir, device):
    cfg = load_hydra_config_from_run(root_dir)
    graph = get_graph(cfg, device)
    noise = get_noise(cfg).to(device)
    score_model = SEDD(cfg).to(device)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=cfg.training.ema)

    ckpt_dir = os.path.join(root_dir, "checkpoints-meta", "checkpoint.pth")
    loaded_state = torch.load(ckpt_dir, map_location=device)

    score_model.load_state_dict(loaded_state["model"])
    ema.load_state_dict(loaded_state["ema"])

    ema.store(score_model.parameters())
    ema.copy_to(score_model.parameters())
    return score_model, graph, noise


def load_model(root_dir, device):
    try:
        return load_model_hf(root_dir, device)
    except Exception as e:
        print(f"Failed to load model from Hugging Face: {e}. Falling back to local.")
        return load_model_local(root_dir, device)
