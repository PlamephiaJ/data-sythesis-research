import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data_process.clean_factory import EmailCleanConfig, EmailCleaner


matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_MODEL_PATH = "exp_local/detection_model/2026.02.23/214447/1"
DEFAULT_INPUT_FILE = "data_phish/eval/Nazario_cleaned_raw.json"


def _safe_max_length(requested: int, tokenizer, model) -> int:
    req = int(requested)
    candidates: List[int] = []

    tok_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(tok_max, int) and 0 < tok_max < 1_000_000:
        candidates.append(tok_max)

    base_model = model.module if hasattr(model, "module") else model
    cfg = getattr(base_model, "config", None)
    model_max_pos = (
        getattr(cfg, "max_position_embeddings", None) if cfg is not None else None
    )
    if isinstance(model_max_pos, int) and model_max_pos > 0:
        candidates.append(model_max_pos)

    if not candidates:
        return req

    return int(min([req] + candidates))


def _resolve_output_paths(
    output_file: Optional[str],
    summary_file: Optional[str],
    model_dir: Path,
    run_dir: Optional[Path],
) -> Tuple[Path, Path]:
    if run_dir is not None:
        base_dir = run_dir / "eval" / time.strftime("%Y%m%d_%H%M%S")
    else:
        base_dir = model_dir.parent / "eval" / time.strftime("%Y%m%d_%H%M%S")

    output_path = Path(output_file) if output_file else base_dir / "predictions.jsonl"
    summary_path = Path(summary_file) if summary_file else base_dir / "summary.json"
    return output_path, summary_path


def _collect_hard_cases(
    results: List[Dict],
    low_conf_threshold: float,
) -> Tuple[List[Dict], List[Dict]]:
    low_confidence: List[Dict] = []
    misclassified: List[Dict] = []

    for record in results:
        pred_confidence = float(record.get("pred_confidence", 0.0))
        true_label = record.get("true_label")
        pred_label = record.get("pred_label")

        if pred_confidence < low_conf_threshold:
            low_confidence.append(record)

        if true_label is not None and int(pred_label) != int(true_label):
            misclassified.append(record)

    return low_confidence, misclassified


def _extract_label(example: Dict, label_key: Optional[str] = None) -> Optional[int]:
    if label_key:
        val = example.get(label_key)
    elif "phish" in example:
        val = example.get("phish")
    elif "label" in example:
        val = example.get("label")
    else:
        return None
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _extract_caption(example: Dict, caption_key: Optional[str] = None) -> Optional[str]:
    if caption_key:
        val = example.get(caption_key)
        if val is not None and str(val).strip():
            return str(val)
    for key in ("style_caption", "caption", "subject"):
        val = example.get(key)
        if val is not None and str(val).strip():
            return str(val)
    return None


def _resolve_model_and_run_dir(model_path: str) -> Tuple[Path, Optional[Path]]:
    path = Path(model_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    if (path / "config.json").exists() and (path / "tokenizer_config.json").exists():
        model_dir = path
        run_dir = (
            path.parent if (path.parent / ".hydra" / "config.yaml").exists() else None
        )
        return model_dir, run_dir

    maybe_model = path / "model"
    if (maybe_model / "config.json").exists() and (
        maybe_model / "tokenizer_config.json"
    ).exists():
        model_dir = maybe_model
        run_dir = path
        return model_dir, run_dir

    raise FileNotFoundError(
        f"Cannot find HuggingFace model files under {path}. "
        f"Expected either <path>/config.json or <path>/model/config.json"
    )


def _load_cleaner_and_max_len(
    run_dir: Optional[Path], default_max_length: int
) -> Tuple[EmailCleaner, int]:
    if run_dir is None:
        return EmailCleaner(EmailCleanConfig()), default_max_length

    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        return EmailCleaner(EmailCleanConfig()), default_max_length

    cfg = OmegaConf.load(str(cfg_path))
    cleaner_cfg = getattr(cfg, "cleaner", None)
    data_cfg = getattr(cfg, "data", None)

    if cleaner_cfg is not None:
        cleaner_dict = OmegaConf.to_container(cleaner_cfg, resolve=True)
        cleaner = EmailCleaner(EmailCleanConfig(**cleaner_dict))
    else:
        cleaner = EmailCleaner(EmailCleanConfig())

    if data_cfg is not None and getattr(data_cfg, "max_length", None) is not None:
        max_length = int(data_cfg.max_length)
    else:
        max_length = default_max_length

    return cleaner, max_length


def _load_samples(input_text: Optional[str], input_file: Optional[str]) -> List[Dict]:
    if input_text is not None:
        return [{"text": input_text}]

    if not input_file:
        raise ValueError("Provide --input_file or --input_text")

    path = Path(input_file).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
                else:
                    records.append({"text": str(obj)})
        return records

    with open(path, "r", encoding="utf-8") as f:
        try:
            obj = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            records = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    records.append(item)
                else:
                    records.append({"text": str(item)})
            if records:
                return records
            raise

    if isinstance(obj, list):
        records = []
        for item in obj:
            if isinstance(item, dict):
                records.append(item)
            else:
                records.append({"text": str(item)})
        return records
    if isinstance(obj, dict):
        return [obj]

    raise ValueError("Input JSON must be a dict, list, or JSONL of dict records")


def _prepare_features(
    samples: List[Dict],
    cleaner: EmailCleaner,
    text_key: str,
    caption_key: Optional[str],
    label_key: Optional[str],
) -> Tuple[List[Dict], List[str], List[Optional[int]]]:
    kept_samples: List[Dict] = []
    cleaned_texts: List[str] = []
    labels: List[Optional[int]] = []

    for item in samples:
        if not isinstance(item, dict):
            continue
        text = item.get(text_key)
        if text is None or not str(text).strip():
            continue

        caption = _extract_caption(item, caption_key)
        cleaned, reason = cleaner.render(caption, str(text))
        if cleaned is None:
            cleaned = (
                f"Subject: {cleaner.cfg.default_subject}\n\n"
                f"{cleaner.cfg.body_prefix}[INVALID SAMPLE: {reason or 'unknown'}]\n"
            )

        kept_samples.append(item)
        cleaned_texts.append(cleaned)
        labels.append(_extract_label(item, label_key))

    if not cleaned_texts:
        raise ValueError("No valid samples to evaluate after filtering")

    return kept_samples, cleaned_texts, labels


def _predict(
    model,
    tokenizer,
    texts: List[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    logits_all: List[torch.Tensor] = []
    model.eval()

    starts = range(0, len(texts), batch_size)
    for start in tqdm(
        starts,
        total=(len(texts) + batch_size - 1) // batch_size,
        desc="Inference",
        unit="batch",
    ):
        batch_texts = texts[start : start + batch_size]
        enc = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = model(**enc)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        logits_all.append(logits.detach().cpu())

    return torch.cat(logits_all, dim=0).numpy()


def _build_summary(
    logits: np.ndarray, labels: List[Optional[int]], threshold: float
) -> Dict:
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    phish_prob = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
    pred_labels = (phish_prob >= threshold).astype(int)

    summary: Dict = {
        "num_samples": int(len(pred_labels)),
        "threshold": float(threshold),
        "predicted_phish_count": int(np.sum(pred_labels == 1)),
        "predicted_ham_count": int(np.sum(pred_labels == 0)),
        "predicted_phish_rate": float(np.mean(pred_labels == 1)),
        "mean_phish_probability": float(np.mean(phish_prob)),
        "std_phish_probability": float(np.std(phish_prob)),
    }

    valid_idx = [i for i, y in enumerate(labels) if y is not None]
    if valid_idx:
        y_true = np.array([int(labels[i]) for i in valid_idx], dtype=int)
        y_prob = phish_prob[valid_idx]
        y_pred = pred_labels[valid_idx]
        summary["label_metrics"] = {
            "num_labeled": int(len(valid_idx)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }
        try:
            summary["label_metrics"]["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            summary["label_metrics"]["roc_auc"] = float("nan")

    return summary


def _save_jsonl(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _save_confusion_matrix_figure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
) -> np.ndarray:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["ham(0)", "phish(1)"],
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return cm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--input_text", type=str, default=None)
    parser.add_argument("--input_file", type=str, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--text_key", type=str, default="text")
    parser.add_argument("--caption_key", type=str, default=None)
    parser.add_argument("--label_key", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=0,
        help="Number of GPUs for parallel inference. 0 means use all available GPUs.",
    )
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--summary_file", type=str, default=None)
    parser.add_argument("--confusion_matrix_file", type=str, default=None)
    parser.add_argument("--misclassified_file", type=str, default=None)
    parser.add_argument("--low_confidence_file", type=str, default=None)
    parser.add_argument("--low_conf_threshold", type=float, default=0.60)
    return parser.parse_args()


def _build_infer_model(
    model_dir: Path,
    device_arg: str,
    num_gpus: int,
):
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))

    if device_arg == "auto":
        if torch.cuda.is_available():
            available_gpus = torch.cuda.device_count()
            use_gpus = (
                available_gpus if num_gpus <= 0 else min(num_gpus, available_gpus)
            )
            device = torch.device("cuda:0")
            model = model.to(device)
            if use_gpus > 1:
                model = torch.nn.DataParallel(model, device_ids=list(range(use_gpus)))
            return model, device, use_gpus

        device = torch.device("cpu")
        model = model.to(device)
        return model, device, 0

    device = torch.device(device_arg)
    model = model.to(device)

    if device.type == "cuda":
        available_gpus = torch.cuda.device_count()
        if available_gpus > 1 and (num_gpus <= 0 or num_gpus > 1):
            use_gpus = (
                available_gpus if num_gpus <= 0 else min(num_gpus, available_gpus)
            )
            model = torch.nn.DataParallel(model, device_ids=list(range(use_gpus)))
            return model, torch.device("cuda:0"), use_gpus
        return model, device, 1

    return model, device, 0


def main() -> None:
    args = parse_args()

    model_dir, run_dir = _resolve_model_and_run_dir(args.model_path)
    cleaner, cfg_max_length = _load_cleaner_and_max_len(run_dir, args.max_length)
    max_length = int(args.max_length) if args.max_length > 0 else int(cfg_max_length)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model, device, used_gpus = _build_infer_model(
        model_dir=model_dir,
        device_arg=args.device,
        num_gpus=args.num_gpus,
    )

    used_max_length = _safe_max_length(max_length, tokenizer, model)
    if int(used_max_length) != int(max_length):
        print(
            f"Warning: requested max_length={int(max_length)} but model/tokenizer supports at most {int(used_max_length)}; clamping.",
            flush=True,
        )
        max_length = int(used_max_length)

    samples = _load_samples(args.input_text, args.input_file)
    samples, cleaned_texts, labels = _prepare_features(
        samples=samples,
        cleaner=cleaner,
        text_key=args.text_key,
        caption_key=args.caption_key,
        label_key=args.label_key,
    )

    logits = _predict(
        model=model,
        tokenizer=tokenizer,
        texts=cleaned_texts,
        max_length=max_length,
        batch_size=args.batch_size,
        device=device,
    )

    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    phish_prob = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
    pred_labels = (phish_prob >= args.threshold).astype(int)

    results = []
    for idx, sample in enumerate(samples):
        results.append(
            {
                "index": idx,
                "pred_label": int(pred_labels[idx]),
                "pred_confidence": float(max(phish_prob[idx], 1 - phish_prob[idx])),
                "phish_probability": float(phish_prob[idx]),
                "true_label": labels[idx],
                "sample": sample,
            }
        )

    summary = _build_summary(logits=logits, labels=labels, threshold=args.threshold)
    summary["model_dir"] = str(model_dir)
    summary["run_dir"] = str(run_dir) if run_dir is not None else None
    summary["device"] = str(device)
    summary["num_gpus_used"] = int(used_gpus)
    summary["max_length"] = int(max_length)
    summary["cleaner_config"] = asdict(cleaner.cfg)

    output_path, summary_path = _resolve_output_paths(
        output_file=args.output_file,
        summary_file=args.summary_file,
        model_dir=model_dir,
        run_dir=run_dir,
    )

    low_confidence, misclassified = _collect_hard_cases(
        results=results,
        low_conf_threshold=float(args.low_conf_threshold),
    )

    low_conf_path = (
        Path(args.low_confidence_file)
        if args.low_confidence_file
        else summary_path.parent / "low_confidence.jsonl"
    )
    misclassified_path = (
        Path(args.misclassified_file)
        if args.misclassified_file
        else summary_path.parent / "misclassified.jsonl"
    )

    _save_jsonl(low_conf_path, low_confidence)
    _save_jsonl(misclassified_path, misclassified)

    summary["hard_case"] = {
        "low_conf_threshold": float(args.low_conf_threshold),
        "low_confidence_count": int(len(low_confidence)),
        "misclassified_count": int(len(misclassified)),
        "low_confidence_file": str(low_conf_path),
        "misclassified_file": str(misclassified_path),
    }

    valid_idx = [i for i, y in enumerate(labels) if y is not None]
    if valid_idx:
        y_true = np.array([int(labels[i]) for i in valid_idx], dtype=int)
        y_pred = pred_labels[valid_idx].astype(int)
        cm_path = (
            Path(args.confusion_matrix_file)
            if args.confusion_matrix_file
            else summary_path.parent / "confusion_matrix.png"
        )
        cm = _save_confusion_matrix_figure(
            y_true=y_true, y_pred=y_pred, save_path=cm_path
        )
        summary.setdefault("label_metrics", {})
        summary["label_metrics"]["confusion_matrix"] = {
            "labels": [0, 1],
            "matrix": cm.tolist(),
            "figure_file": str(cm_path),
        }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    _save_jsonl(output_path, results)
    print(f"Saved predictions to: {output_path}")
    print(f"Saved low-confidence cases to: {low_conf_path}")
    print(f"Saved misclassified cases to: {misclassified_path}")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
