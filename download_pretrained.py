#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"


DEFAULT_CONFIG_PATHS = (
    REPO_ROOT / "configs" / "config.yaml",
    REPO_ROOT / "src" / "detection_model" / "config_detect.yaml",
)
PROJECT_MODEL_ALIASES = {"small", "medium", "small-raw"}


def _collect_models_from_config(config_path: Path) -> set[str]:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config_path)
    models: set[str] = set()

    tokenizer_cfg = cfg.get("tokenizer")
    if tokenizer_cfg:
        text_name = tokenizer_cfg.get("text")
        caption_name = tokenizer_cfg.get("caption")
        if text_name:
            models.add(str(text_name))
        if caption_name:
            models.add(str(caption_name))

    pretrained_cfg = cfg.get("pretrained")
    if pretrained_cfg:
        alignment_name = pretrained_cfg.get("alignment_model")
        perplexity_name = pretrained_cfg.get("perplexity_model")
        if alignment_name:
            models.add(str(alignment_name))
        if perplexity_name:
            models.add(str(perplexity_name))

    model_cfg = cfg.get("model")
    if model_cfg:
        model_name = model_cfg.get("name")
        if model_name:
            model_name = str(model_name)
            is_local_path = (
                Path(model_name).is_absolute() or (REPO_ROOT / model_name).exists()
            )
            if model_name not in PROJECT_MODEL_ALIASES and not is_local_path:
                models.add(model_name)
        caption_encoder_cfg = model_cfg.get("caption_encoder")
        if caption_encoder_cfg and caption_encoder_cfg.get("name"):
            models.add(str(caption_encoder_cfg.get("name")))

    return models


def _iter_default_models(config_paths: Iterable[Path]) -> list[str]:
    models: set[str] = set()
    for config_path in config_paths:
        if not config_path.exists():
            continue
        models.update(_collect_models_from_config(config_path))
    return sorted(models)


def main() -> None:
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    from utils import hf_local

    parser = argparse.ArgumentParser(
        description="Pre-download pretrained Hugging Face models into the local hf cache."
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Hugging Face model id to download. Can be passed multiple times.",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Extra config file to scan for pretrained model ids. Can be passed multiple times.",
    )
    parser.add_argument(
        "--local-dir",
        default="hf-local",
        help="Relative cache directory under repo root. Default: hf-local",
    )
    parser.add_argument(
        "--skip-default-configs",
        action="store_true",
        help="Do not scan the default project configs for pretrained model ids.",
    )
    args = parser.parse_args()

    config_paths = []
    if not args.skip_default_configs:
        config_paths.extend(DEFAULT_CONFIG_PATHS)
    config_paths.extend(REPO_ROOT / Path(path) for path in args.config)

    models = set(_iter_default_models(config_paths))
    models.update(str(model) for model in args.model if str(model).strip())
    models = {model for model in models if model and str(model).strip()}

    if not models:
        raise SystemExit(
            "No pretrained models found. Pass --model or provide config files."
        )

    hf_local.configure(local_dir=args.local_dir, download_if_missing=True)

    print(f"Cache dir: {REPO_ROOT / args.local_dir}")
    for model_name in sorted(models):
        local_path = hf_local.resolve_pretrained_path(model_name)
        print(f"{model_name} -> {local_path}")


if __name__ == "__main__":
    main()
