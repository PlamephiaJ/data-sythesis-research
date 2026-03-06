import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from utils.tokenizer_factory import get_caption_tokenizer


FILE_RE = re.compile(r"step_(\d+)_block_(\d+)\.pt$")


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_tokens(caption: str, tokenizer_name: str, style_len: int) -> List[str]:
    tokenizer = get_caption_tokenizer(tokenizer_name)
    encoded = tokenizer(caption, add_special_tokens=True)
    token_ids = encoded["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    if len(tokens) > style_len:
        tokens = tokens[:style_len]
    elif len(tokens) < style_len:
        tokens = tokens + ["<pad>"] * (style_len - len(tokens))
    return tokens


def _collect_attn_files(sample_dir: Path) -> Dict[int, Dict[int, Path]]:
    step_to_blocks: Dict[int, Dict[int, Path]] = {}
    for p in sorted(sample_dir.glob("step_*_block_*.pt")):
        m = FILE_RE.search(p.name)
        if m is None:
            continue
        step_idx = int(m.group(1))
        block_idx = int(m.group(2))
        step_to_blocks.setdefault(step_idx, {})[block_idx] = p
    return step_to_blocks


def _pick_three(values: List[int]) -> List[int]:
    if len(values) == 0:
        raise ValueError("Empty index list cannot pick 3 points")
    if len(values) == 1:
        return [values[0], values[0], values[0]]
    if len(values) == 2:
        return [values[0], values[1], values[1]]
    return [values[0], values[len(values) // 2], values[-1]]


def _load_head_avg_attn(attn_file: Path) -> torch.Tensor:
    attn = torch.load(attn_file, map_location="cpu")
    if attn.ndim != 3:
        raise ValueError(
            f"Expected [heads, x_seq_len, style_seq_len], got {tuple(attn.shape)}"
        )
    return attn.float().mean(dim=0)


def _plot_3x3(
    panels: List[List[torch.Tensor]],
    step_indices: List[int],
    block_indices: List[int],
    caption_tokens: List[str],
    out_file: Path,
):
    fig, axes = plt.subplots(3, 3, figsize=(18, 12), dpi=150)
    last_im = None

    if len(caption_tokens) <= 32:
        x_ticks = list(range(len(caption_tokens)))
    else:
        stride = max(1, len(caption_tokens) // 16)
        x_ticks = list(range(0, len(caption_tokens), stride))

    for i in range(3):
        for j in range(3):
            ax = axes[i][j]
            mat = panels[i][j]
            last_im = ax.imshow(
                mat.cpu().numpy(),
                aspect="auto",
                interpolation="nearest",
                origin="lower",
            )
            ax.set_title(f"step={step_indices[i]}, block={block_indices[j]}")
            ax.set_xlabel("caption tokens")
            ax.set_ylabel("text positions")
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(
                [caption_tokens[k] for k in x_ticks],
                rotation=75,
                ha="right",
                fontsize=8,
            )

            y_len = mat.size(0)
            if y_len <= 20:
                y_ticks = list(range(y_len))
            else:
                y_stride = max(1, y_len // 10)
                y_ticks = list(range(0, y_len, y_stride))
            ax.set_yticks(y_ticks)

    if last_im is not None:
        fig.colorbar(last_im, ax=axes, fraction=0.015, pad=0.01)
    fig.suptitle(
        "Cross-attn heatmaps (head-avg): early/mid/late × low/mid/high", y=0.995
    )
    fig.tight_layout(rect=[0, 0, 0.98, 0.98])
    fig.savefig(out_file)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize saved cross-attention maps from sampling."
    )
    parser.add_argument("--sample_json", type=str, required=True)
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--attn_root", type=str, default=None)
    parser.add_argument("--caption_tokenizer", type=str, default="bert-base-uncased")
    args = parser.parse_args()

    sample_json = Path(args.sample_json)
    sample_parent = sample_json.parent
    if args.attn_root is None:
        stem = sample_json.stem
        attn_root = sample_parent / f"{stem}_cross_attn_maps"
    else:
        attn_root = Path(args.attn_root)

    sample_dir = attn_root / f"sample_{args.sample_idx}"
    out_dir = sample_dir / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not sample_json.exists():
        raise FileNotFoundError(f"sample_json not found: {sample_json}")
    if not sample_dir.exists():
        raise FileNotFoundError(f"attention map dir not found: {sample_dir}")

    samples = _read_json(sample_json)
    if args.sample_idx < 0 or args.sample_idx >= len(samples):
        raise IndexError(
            f"sample_idx {args.sample_idx} out of range [0, {len(samples) - 1}]"
        )

    caption = samples[args.sample_idx].get("caption", "")

    step_to_blocks = _collect_attn_files(sample_dir)
    if len(step_to_blocks) == 0:
        raise RuntimeError(f"No step/block attention files found under: {sample_dir}")

    first_step = min(step_to_blocks.keys())
    first_block = min(step_to_blocks[first_step].keys())
    first_attn = torch.load(step_to_blocks[first_step][first_block], map_location="cpu")
    if first_attn.ndim != 3:
        raise ValueError(
            f"Expected attention tensor shape [heads, x_seq_len, style_seq_len], got {tuple(first_attn.shape)}"
        )
    style_len = int(first_attn.shape[-1])

    tokens = _build_tokens(caption, args.caption_tokenizer, style_len)

    available_steps = sorted(step_to_blocks.keys())
    common_blocks = set(step_to_blocks[available_steps[0]].keys())
    for step_idx in available_steps[1:]:
        common_blocks &= set(step_to_blocks[step_idx].keys())
    if len(common_blocks) == 0:
        raise RuntimeError("No common block indices across all saved timesteps")

    available_blocks = sorted(common_blocks)
    selected_steps = _pick_three(available_steps)
    selected_blocks = _pick_three(available_blocks)

    panels: List[List[torch.Tensor]] = []
    for step_idx in selected_steps:
        row_panels: List[torch.Tensor] = []
        for block_idx in selected_blocks:
            attn_file = step_to_blocks[step_idx][block_idx]
            row_panels.append(_load_head_avg_attn(attn_file))
        panels.append(row_panels)

    heatmap_file = out_dir / "cross_attn_3x3.png"
    _plot_3x3(
        panels=panels,
        step_indices=selected_steps,
        block_indices=selected_blocks,
        caption_tokens=tokens,
        out_file=heatmap_file,
    )

    summary = {
        "sample_json": str(sample_json),
        "attn_root": str(attn_root),
        "sample_idx": int(args.sample_idx),
        "selected_steps": selected_steps,
        "selected_blocks": selected_blocks,
        "num_available_steps": int(len(available_steps)),
        "num_available_blocks": int(len(available_blocks)),
        "style_len": style_len,
        "caption": caption,
        "outputs": {
            "heatmap_3x3": str(heatmap_file),
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
