#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compute dataset and sequence-length statistics.

Supports both data.format = "entry" and "chunk" configs.

Outputs per split:
    - total examples / batches processed
    - total tokens (non-pad), mean/std, min/max
    - padding rate (requires fixed max length)
    - approximate quantiles from a bounded reservoir sample
"""

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data_process.data import get_chunk_dataset, get_entry_dataset


def _extract_lengths(
    batch: Dict[str, torch.Tensor],
) -> Dict[str, Tuple[torch.Tensor, Optional[int]]]:
    """Return per-field (lengths, max_len_if_known).

    lengths are number of valid (non-pad) tokens per example.
    """

    out: Dict[str, Tuple[torch.Tensor, Optional[int]]] = {}

    if "text_attention_mask" in batch:
        mask = batch["text_attention_mask"]
        out["text"] = (mask.sum(dim=1).to(torch.long), int(mask.shape[1]))
    elif "input_ids" in batch:
        # chunked LM dataset: fixed-size blocks
        ids = batch["input_ids"]
        out["text"] = (
            torch.full(
                (ids.shape[0],),
                ids.shape[1],
                device=ids.device,
                dtype=torch.long,
            ),
            int(ids.shape[1]),
        )

    if "style_caption_attention_mask" in batch:
        mask = batch["style_caption_attention_mask"]
        out["caption"] = (mask.sum(dim=1).to(torch.long), int(mask.shape[1]))

    return out


@dataclass
class LengthStats:
    count: int = 0
    tok_sum: int = 0
    tok_sum_sq: float = 0.0
    mn: int = 2**31 - 1
    mx: int = 0
    max_len: Optional[int] = None
    # bounded reservoir sampling for approximate quantiles
    sample: Optional[torch.Tensor] = None  # 1D CPU long
    seen: int = 0


def _reservoir_update(
    cur: Optional[torch.Tensor],
    incoming: torch.Tensor,
    *,
    reservoir_size: int,
    seen: int,
) -> Tuple[torch.Tensor, int]:
    """Reservoir-sample incoming 1D CPU tensor into a fixed-size buffer."""
    if reservoir_size <= 0:
        # keep nothing
        return torch.empty((0,), dtype=torch.long), seen + int(incoming.numel())

    incoming = incoming.to(dtype=torch.long, device="cpu").flatten()
    n_in = int(incoming.numel())
    if n_in == 0:
        return cur if cur is not None else torch.empty((0,), dtype=torch.long), seen

    if cur is None or cur.numel() == 0:
        take = min(reservoir_size, n_in)
        return incoming[:take].clone(), seen + n_in

    cur = cur.clone()
    cur_n = int(cur.numel())

    # Fill up if not full.
    if cur_n < reservoir_size:
        take = min(reservoir_size - cur_n, n_in)
        cur = torch.cat([cur, incoming[:take]], dim=0)
        incoming = incoming[take:]
        n_in = int(incoming.numel())
        cur_n = int(cur.numel())

    # Replace with decreasing probability.
    if n_in > 0 and cur_n > 0:
        # For each incoming element i, choose j ~ [0, seen+i]. If j < reservoir_size, replace.
        # Vectorized approximate implementation.
        base = seen + (torch.arange(n_in, dtype=torch.long) + 1)
        j = torch.randint(
            low=0,
            high=int(base[-1].item()) + 1,
            size=(n_in,),
            dtype=torch.long,
        )
        hit = j < reservoir_size
        if hit.any():
            cur[j[hit]] = incoming[hit]

    return cur, seen + n_in


def compute_dataset_stats(
    loader: DataLoader,
    *,
    max_batches: int = -1,
    reservoir_size: int = 50000,
) -> Dict[str, Dict[str, float]]:
    per_key: Dict[str, LengthStats] = {}
    num_batches = 0
    num_examples = 0

    for i, batch in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        num_batches += 1

        batch_lengths = _extract_lengths(batch)
        if not batch_lengths:
            continue

        # infer batch size from first entry
        first_lengths, _ = next(iter(batch_lengths.values()))
        num_examples += int(first_lengths.numel())

        for k, (lengths, max_len) in batch_lengths.items():
            lengths_cpu = lengths.detach().to(device="cpu", dtype=torch.long)
            st = per_key.get(k)
            if st is None:
                st = LengthStats()
                per_key[k] = st

            if max_len is not None:
                if st.max_len is None:
                    st.max_len = int(max_len)
                elif st.max_len != int(max_len):
                    raise ValueError(
                        f"Non-constant max_len for {k}: saw {st.max_len} then {max_len}"
                    )

            st.count += int(lengths_cpu.numel())
            st.tok_sum += int(lengths_cpu.sum().item())
            st.tok_sum_sq += float((lengths_cpu.to(torch.float64) ** 2).sum().item())
            st.mn = min(st.mn, int(lengths_cpu.min().item()))
            st.mx = max(st.mx, int(lengths_cpu.max().item()))
            st.sample, st.seen = _reservoir_update(
                st.sample,
                lengths_cpu,
                reservoir_size=reservoir_size,
                seen=st.seen,
            )

    out: Dict[str, Dict[str, float]] = {
        "__meta__": {
            "num_batches": float(num_batches),
            "num_examples": float(num_examples),
        }
    }

    for k, st in per_key.items():
        if st.count <= 0:
            continue

        mean = st.tok_sum / st.count
        var = max(0.0, (st.tok_sum_sq / st.count) - (mean**2))
        std = var**0.5

        sample = st.sample
        if sample is not None and sample.numel() > 0:
            sample_f = sample.to(torch.float32)
            q = torch.quantile(
                sample_f,
                torch.tensor([0.25, 0.50, 0.75, 0.90, 0.95, 0.99]),
            )
            q25, q50, q75, q90, q95, q99 = [float(x.item()) for x in q]
        else:
            q25 = q50 = q75 = q90 = q95 = q99 = 0.0

        pad_rate = 0.0
        if st.max_len is not None and st.max_len > 0:
            pad_rate = 1.0 - (st.tok_sum / (st.count * st.max_len))

        out[k] = {
            "count": float(st.count),
            "total_tokens": float(st.tok_sum),
            "mean": float(mean),
            "std": float(std),
            "min": float(st.mn if st.mn != 2**31 - 1 else 0),
            "max": float(st.mx),
            "q25": q25,
            "median": q50,
            "q75": q75,
            "q90": q90,
            "q95": q95,
            "q99": q99,
            "max_len": float(st.max_len) if st.max_len is not None else 0.0,
            "pad_rate": float(pad_rate),
            "sample_n": float(sample.numel()) if sample is not None else 0.0,
        }

    return out


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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to a Hydra-composed config yaml (defaults to configs/config.yaml).",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=-1,
        help="If >0, only iterate this many DataLoader batches.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader num_workers.",
    )
    parser.add_argument(
        "--reservoir-size",
        type=int,
        default=50000,
        help="Sample size for approximate quantiles (memory-bounded).",
    )
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
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
            num_workers=int(args.num_workers),
            pin_memory=True,
        )

        stats = compute_dataset_stats(
            loader,
            max_batches=int(args.max_batches),
            reservoir_size=int(args.reservoir_size),
        )

        meta = stats.get("__meta__", {})
        dataset_len = None
        try:
            dataset_len = len(dataset)
        except Exception:
            dataset_len = None

        header_bits = [f"split={split}"]
        if dataset_len is not None:
            header_bits.append(f"dataset_len={dataset_len}")
        header_bits.append(f"examples={int(meta.get('num_examples', 0))}")
        header_bits.append(f"batches={int(meta.get('num_batches', 0))}")
        print("[dataset] " + ", ".join(header_bits))

        for key in ("text", "caption"):
            if key not in stats:
                continue
            v = stats[key]
            print(
                f"[{split}] {key}: "
                f"total_tokens={v['total_tokens']:.0f}, "
                f"mean={v['mean']:.2f}±{v['std']:.2f}, "
                f"min={v['min']:.0f}, q25={v['q25']:.0f}, median={v['median']:.0f}, q75={v['q75']:.0f}, "
                f"p90={v['q90']:.0f}, p95={v['q95']:.0f}, p99={v['q99']:.0f}, max={v['max']:.0f}, "
                f"max_len={v['max_len']:.0f}, pad_rate={v['pad_rate']*100:.2f}%, sample_n={v['sample_n']:.0f}"
            )


if __name__ == "__main__":
    main()
