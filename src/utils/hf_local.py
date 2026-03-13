from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

from huggingface_hub import snapshot_download


logger = logging.getLogger(__name__)

_DEFAULT_LOCAL_DIR = "hf-local"
_DEFAULT_DOWNLOAD_IF_MISSING = True
_DOWNLOAD_ALLOW_PATTERNS = (
    "*.bin",
    "*.json",
    "*.md",
    "*.model",
    "*.pt",
    "*.py",
    "*.safetensors",
    "*.spm",
    "*.txt",
)
_DOWNLOAD_IGNORE_PATTERNS = (
    "coreml/*",
    "onnx/*",
    "openvino/*",
    "*.h5",
    "*.msgpack",
    "*.onnx",
    "*.ot",
    "*.tflite",
)


def _get_nested(cfg: Any, *keys: str, default: Any = None) -> Any:
    current = cfg
    for key in keys:
        if current is None:
            return default
        try:
            if key not in current:
                return default
            current = current[key]
            continue
        except Exception:
            pass
        try:
            current = getattr(current, key)
        except Exception:
            return default
    return current


def configure(
    *,
    local_dir: Optional[str] = None,
    download_if_missing: Optional[bool] = None,
) -> None:
    global _DEFAULT_LOCAL_DIR
    global _DEFAULT_DOWNLOAD_IF_MISSING

    if local_dir:
        _DEFAULT_LOCAL_DIR = str(local_dir)
    if download_if_missing is not None:
        _DEFAULT_DOWNLOAD_IF_MISSING = bool(download_if_missing)


def configure_from_config(cfg: Any) -> None:
    configure(
        local_dir=_get_nested(cfg, "pretrained", "local_dir", default=None),
        download_if_missing=_get_nested(
            cfg, "pretrained", "download_if_missing", default=None
        ),
    )


def get_repo_root() -> Path:
    current = Path(__file__).resolve().parent
    for parent in (current, *current.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return current.parents[2]


def get_local_model_root(local_dir: Optional[str] = None) -> Path:
    root = get_repo_root() / str(local_dir or _DEFAULT_LOCAL_DIR)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resolve_existing_local_path(model_name_or_path: str) -> Optional[Path]:
    candidate = Path(os.path.expanduser(str(model_name_or_path)))

    if candidate.is_absolute():
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(f"Local pretrained path not found: {candidate}")

    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    repo_candidate = (get_repo_root() / candidate).resolve()
    if repo_candidate.exists():
        return repo_candidate

    return None


def _prune_duplicate_weight_formats(root_dir: Path) -> None:
    safetensor_dirs = {path.parent for path in root_dir.rglob("*.safetensors")}
    for model_dir in safetensor_dirs:
        for bin_file in model_dir.glob("pytorch_model*.bin"):
            logger.info("Removing duplicate PyTorch bin weight '%s'.", bin_file)
            bin_file.unlink(missing_ok=True)


def resolve_pretrained_path(
    model_name_or_path: str,
    *,
    local_dir: Optional[str] = None,
    download_if_missing: Optional[bool] = None,
) -> str:
    existing_path = _resolve_existing_local_path(model_name_or_path)
    if existing_path is not None:
        return str(existing_path)

    target_dir = get_local_model_root(local_dir=local_dir) / str(model_name_or_path)
    if target_dir.exists() and any(target_dir.iterdir()):
        return str(target_dir.resolve())

    should_download = (
        _DEFAULT_DOWNLOAD_IF_MISSING
        if download_if_missing is None
        else bool(download_if_missing)
    )
    if not should_download:
        raise FileNotFoundError(
            f"Pretrained model '{model_name_or_path}' was not found in {target_dir}."
        )

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Downloading pretrained model '%s' into '%s'.",
        model_name_or_path,
        target_dir,
    )
    snapshot_download(
        repo_id=str(model_name_or_path),
        local_dir=str(target_dir),
        allow_patterns=list(_DOWNLOAD_ALLOW_PATTERNS),
        ignore_patterns=list(_DOWNLOAD_IGNORE_PATTERNS),
    )
    _prune_duplicate_weight_formats(target_dir)
    return str(target_dir.resolve())
