#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compute average sequence length from current dataloader datasets.

Supports both data.format = "entry" and "chunk" configs.
"""

import os
from typing import Dict, Tuple

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data_process.data import get_chunk_dataset, get_entry_dataset


def _extract_lengths(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    lengths = {}

    if "text_attention_mask" in batch:
        lengths["text"] = batch["text_attention_mask"].sum(dim=1).to(torch.long)
    elif "input_ids" in batch:
        # chunked LM dataset: fixed-size blocks
        lengths["text"] = torch.full(
            (batch["input_ids"].shape[0],),
            batch["input_ids"].shape[1],
            device=batch["input_ids"].device,
            dtype=torch.long,
        )

    if "style_caption_attention_mask" in batch:
        lengths["caption"] = (
            batch["style_caption_attention_mask"].sum(dim=1).to(torch.long)
        )

    return lengths


def _merge_stats(
    total: Dict[str, Tuple[int, int, int, int]],
    batch: Dict[str, Tuple[int, int, int, int]],
):
    for k, (tok_sum, count, mn, mx) in batch.items():
        if k not in total:
            total[k] = (0, 0, mn, mx)
        total[k] = (
            total[k][0] + tok_sum,
            total[k][1] + count,
            min(total[k][2], mn),
            max(total[k][3], mx),
        )


def compute_avg_lengths(
    loader: DataLoader, max_batches: int = -1
) -> Dict[str, Dict[str, float]]:
    total: Dict[str, Tuple[int, int, int, int]] = {}
    all_lengths: Dict[str, list[torch.Tensor]] = {}

    for i, batch in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        batch_lengths = _extract_lengths(batch)

        batch_stats = {}
        for k, lengths in batch_lengths.items():
            batch_stats[k] = (
                int(lengths.sum().item()),
                int(lengths.numel()),
                int(lengths.min().item()),
                int(lengths.max().item()),
            )
            all_lengths.setdefault(k, []).append(lengths.detach().cpu())

        _merge_stats(total, batch_stats)

    stats: Dict[str, Dict[str, float]] = {}
    for k, (tok_sum, count, mn, mx) in total.items():
        if k in all_lengths and all_lengths[k]:
            concat = torch.cat(all_lengths[k], dim=0)
            median = float(concat.median().item())
        else:
            median = 0.0
        stats[k] = {
            "avg": (tok_sum / count) if count > 0 else 0.0,
            "min": float(mn),
            "max": float(mx),
            "median": median,
        }
    return stats


def build_dataset(cfg, split: str):
    if cfg.data.format == "chunk":
        name = cfg.data.trainset.name if split == "train" else cfg.data.validset.name
        cache_dir = (
            cfg.data.trainset.cache_dir
            if split == "train"
            else cfg.data.validset.cache_dir
        )
        mode = "train" if split == "train" else "validation"
        if name == "text8" and mode == "validation":
            mode = "test"
        return get_chunk_dataset(
            name,
            mode,
            cache_dir=cache_dir,
            block_size=cfg.model.length,
            num_proc=cfg.data.num_proc,
        )

    if cfg.data.format == "entry":
        name = cfg.data.trainset.name if split == "train" else cfg.data.validset.name
        cache_dir = (
            cfg.data.trainset.cache_dir
            if split == "train"
            else cfg.data.validset.cache_dir
        )
        mode = "train" if split == "train" else "validation"
        return get_entry_dataset(
            name,
            mode,
            cache_dir=cache_dir,
            text_max_length=cfg.data.max_length,
            caption_max_length=getattr(cfg.data, "caption_max_length", 256),
            num_proc=cfg.data.num_proc,
            text_tokenizer_name=cfg.tokenizer.text,
            caption_tokenizer_name=cfg.tokenizer.caption,
        )

    raise ValueError(f"Unknown data.format: {cfg.data.format}")


def _load_cfg(config_path: str):
    cfg = OmegaConf.load(config_path)
    if "data" in cfg and "model" in cfg:
        return cfg

    cfg_dir = os.path.abspath(os.path.dirname(config_path))
    cfg_name = os.path.splitext(os.path.basename(config_path))[0]

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        return compose(config_name=cfg_name)


def main():
    config_path = "configs/config.yaml"
    cfg = _load_cfg(config_path)

    splits = ["train", "validation"]

    for split in splits:
        dataset = build_dataset(cfg, split)
        batch_size = (
            cfg.training.batch_size if split == "train" else cfg.eval.batch_size
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        stats = compute_avg_lengths(loader, max_batches=-1)
        stats_items = ", ".join(
            [
                (
                    f"{k}=avg:{v['avg']:.2f}, "
                    f"median:{v['median']:.0f}, "
                    f"min:{v['min']:.0f}, max:{v['max']:.0f}"
                )
                for k, v in stats.items()
            ]
        )
        print(f"[{split}] length_stats: {stats_items}")


if __name__ == "__main__":
    main()
