#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compute dataset-level perplexity (PPL) using the same loss aggregation as
`src/train/run_train.py`.

This script:
  - builds the dataset from a Hydra config (supports data.format = entry/chunk)
  - runs a causal LM (default: gpt2-large) to compute next-token cross-entropy
  - aggregates loss over the whole dataset with token-weighted mean
  - reports PPL = exp(mean_loss)

Notes on masking:
  - For entry datasets (padded), we use attention_mask when available.
  - Optionally (default for entry), we truncate loss at the first EOS token
        in each sequence (matching run_train's sample PPL behavior).
  - For chunk datasets, truncation-at-EOS is disabled by default because
        chunks may contain multiple documents separated by EOS.
"""

import argparse
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data_process.data import get_chunk_dataset, get_entry_dataset
from utils.eval_factory import get_eval_lm
from utils.tokenizer_factory import get_text_tokenizer


def _load_cfg(config_path: str):
    """Load a config.

    Accepts either:
      - a fully composed Hydra yaml that already contains `data` and `model`, or
      - a Hydra root config file (e.g. configs/config.yaml) that needs composition.
    """

    cfg = OmegaConf.load(config_path)
    if "data" in cfg and "model" in cfg:
        return cfg

    cfg_dir = os.path.abspath(os.path.dirname(config_path))
    cfg_name = os.path.splitext(os.path.basename(config_path))[0]

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        return compose(config_name=cfg_name)


def build_dataset(cfg, split: str):
    if split not in {"train", "validation"}:
        raise ValueError(f"Unknown split: {split}. Expected train|validation")

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


def _extract_text_fields(
    batch: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return (input_ids, attention_mask_or_None) for the text stream."""

    if "text_input_ids" in batch:
        input_ids = batch["text_input_ids"]
        attention_mask = batch.get("text_attention_mask")
        return input_ids, attention_mask
    if "input_ids" in batch:
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        return input_ids, attention_mask
    raise KeyError(f"Cannot find text tokens in batch keys: {sorted(batch.keys())}")


@torch.inference_mode()
def compute_dataset_ppl(
    loader: DataLoader,
    *,
    lm_name: str,
    device: torch.device,
    eos_token_id: int,
    truncate_at_eos: bool,
    use_attention_mask: bool,
    log_every: int,
    use_amp: bool,
) -> Dict[str, float]:
    eval_model = get_eval_lm(lm_name, str(device))
    eval_model.eval()

    total_loss = torch.zeros(1, device=device, dtype=torch.float64)
    total_tokens = torch.zeros(1, device=device, dtype=torch.float64)

    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.float16)
        if (use_amp and device.type == "cuda")
        else torch.autocast(device_type="cpu", enabled=False)
    )

    for step, batch in enumerate(loader, start=1):
        input_ids_cpu, attention_mask_cpu = _extract_text_fields(batch)
        input_ids = input_ids_cpu.to(device, non_blocking=True)
        attention_mask = (
            attention_mask_cpu.to(device, non_blocking=True)
            if (use_attention_mask and attention_mask_cpu is not None)
            else None
        )

        if input_ids.ndim != 2 or input_ids.size(1) < 2:
            continue

        with autocast_ctx:
            outputs = eval_model(input_ids)
            logits = outputs.logits  # (b, T, V)

            logits = logits[:, :-1, :].contiguous()  # (b, T-1, V)
            targets = input_ids[:, 1:]  # (b, T-1)

            vocab_size = logits.size(-1)
            token_losses = F.cross_entropy(
                logits.view(-1, vocab_size),
                targets.reshape(-1),
                reduction="none",
            ).view_as(
                targets
            )  # (b, T-1)

            valid_mask = torch.ones_like(token_losses, dtype=token_losses.dtype)

            if attention_mask is not None:
                valid_mask = valid_mask * attention_mask[:, 1:].to(token_losses.dtype)

            if truncate_at_eos:
                eos_mask = (targets == eos_token_id).int()
                cumsum = eos_mask.cumsum(dim=-1)
                valid_mask = valid_mask * (cumsum == 0).to(token_losses.dtype)

            masked_losses = token_losses * valid_mask
            total_loss += masked_losses.sum().to(torch.float64)
            total_tokens += valid_mask.sum().to(torch.float64)

        if log_every > 0 and step % log_every == 0:
            tok = float(total_tokens.clamp_min(1).item())
            mean_loss = float((total_loss / total_tokens.clamp_min(1)).item())
            ppl = float(torch.exp(torch.tensor(mean_loss)).item())
            print(
                f"[progress] batches={step}, tokens={tok:.0f}, mean_loss={mean_loss:.6f}, ppl={ppl:.3f}"
            )

    total_tokens = total_tokens.clamp_min(1)
    mean_loss = (total_loss / total_tokens).to(torch.float64)
    ppl = torch.exp(mean_loss)

    return {
        "total_tokens": float(total_tokens.item()),
        "mean_loss": float(mean_loss.item()),
        "ppl": float(ppl.item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to a Hydra-composed config yaml (defaults to configs/config.yaml).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation"],
        help="Which dataset split to evaluate.",
    )
    parser.add_argument(
        "--lm",
        type=str,
        default="gpt2-large",
        help="HuggingFace causal LM name (default: gpt2-large).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=-1,
        help="Override batch size (default: cfg.eval.perplexity_batch_size).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=120,
        help="DataLoader num_workers.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=-1,
        help="If >0, only iterate this many DataLoader batches.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run eval LM on.",
    )
    parser.add_argument(
        "--truncate-at-eos",
        type=int,
        default=-1,
        help="1 to truncate loss at first EOS, 0 to disable. Default: entry=1, chunk=0.",
    )
    parser.add_argument(
        "--no-attention-mask",
        action="store_true",
        help="Disable using attention_mask even if present.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print running stats every N batches (0 disables).",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable AMP autocast on CUDA.",
    )
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    dataset = build_dataset(cfg, args.split)

    default_bs = int(getattr(cfg.eval, "perplexity_batch_size", 32))
    batch_size = int(args.batch_size) if int(args.batch_size) > 0 else default_bs

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    truncate_default = True if cfg.data.format == "entry" else False
    if int(args.truncate_at_eos) == 0:
        truncate_at_eos = False
    elif int(args.truncate_at_eos) == 1:
        truncate_at_eos = True
    else:
        truncate_at_eos = truncate_default

    tokenizer = get_text_tokenizer(cfg.tokenizer.text)
    eos_token_id = int(tokenizer.eos_token_id)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    if int(args.max_batches) > 0:
        # Wrap loader to stop early without materializing data.
        def _limited():
            for i, b in enumerate(loader):
                if i >= int(args.max_batches):
                    break
                yield b

        iter_loader = _limited()
    else:
        iter_loader = loader

    dataset_len = None
    try:
        dataset_len = len(dataset)
    except Exception:
        dataset_len = None

    header = [
        f"split={args.split}",
        f"format={cfg.data.format}",
        f"dataset={cfg.data.trainset.name if args.split=='train' else cfg.data.validset.name}",
        f"lm={args.lm}",
        f"batch_size={batch_size}",
        f"device={device}",
        f"truncate_at_eos={int(truncate_at_eos)}",
    ]
    if dataset_len is not None:
        header.append(f"dataset_len={dataset_len}")
    print("[dataset_ppl] " + ", ".join(header))

    stats = compute_dataset_ppl(
        iter_loader,
        lm_name=args.lm,
        device=device,
        eos_token_id=eos_token_id,
        truncate_at_eos=truncate_at_eos,
        use_attention_mask=(not args.no_attention_mask),
        log_every=int(args.log_every),
        use_amp=(not args.no_amp),
    )

    print(
        f"[result] total_tokens={stats['total_tokens']:.0f}, mean_loss={stats['mean_loss']:.6f}, ppl={stats['ppl']:.3f}"
    )


if __name__ == "__main__":
    main()
