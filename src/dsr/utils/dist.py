"""Distributed training and checkpoint helpers."""

from __future__ import annotations

import logging
import os

import torch
from omegaconf import OmegaConf

from .logging import makedirs


def load_hydra_config_from_run(load_dir: str):
    cfg_path = os.path.join(load_dir, ".hydra", "config.yaml")
    return OmegaConf.load(cfg_path)


def restore_checkpoint(ckpt_path: str, state: dict, device: torch.device) -> dict:
    if not os.path.exists(ckpt_path):
        makedirs(os.path.dirname(ckpt_path))
        logging.warning("No checkpoint found at %s. Returned the same state as input", ckpt_path)
        return state

    loaded_state = torch.load(ckpt_path, map_location=device)
    state["optimizer"].load_state_dict(loaded_state["optimizer"])
    state["model"].module.load_state_dict(loaded_state["model"], strict=False)
    state["ema"].load_state_dict(loaded_state["ema"])
    state["step"] = loaded_state["step"]
    return state


def save_checkpoint(ckpt_path: str, state: dict) -> None:
    saved_state = {
        "optimizer": state["optimizer"].state_dict(),
        "model": state["model"].module.state_dict(),
        "ema": state["ema"].state_dict(),
        "step": state["step"],
    }
    torch.save(saved_state, ckpt_path)
