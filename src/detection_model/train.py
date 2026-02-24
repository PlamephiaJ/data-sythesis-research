import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
import torch
import torch.multiprocessing as mp
from datasets import Dataset, concatenate_datasets, load_dataset
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    set_seed,
)

from data_process import dataset_factory
from data_process.clean_factory import EmailCleanConfig, EmailCleaner


def build_cleaner(cfg: DictConfig) -> EmailCleaner:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    return EmailCleaner(EmailCleanConfig(**cfg_dict))


def _extract_label(example: Dict) -> Optional[int]:
    if "phish" in example:
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


def _extract_caption(example: Dict) -> Optional[str]:
    for key in ("style_caption", "caption", "subject"):
        val = example.get(key)
        if val is not None and str(val).strip():
            return str(val)
    return None


def _load_json_data(data_path: str) -> Dataset:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data path not found: {path}")

    if path.is_file():
        if path.suffix.lower() not in {".json", ".jsonl"}:
            raise ValueError(f"Unsupported data file type: {path}")
        data_files = [str(path)]
    else:
        json_files = sorted(path.glob("*.json")) + sorted(path.glob("*.jsonl"))
        if not json_files:
            raise FileNotFoundError(f"No JSON/JSONL files found in {path}")
        data_files = [str(p) for p in json_files]

    ds = load_dataset("json", data_files=data_files)["train"]
    return ds


def _prepare_dataset(
    ds: Dataset,
    tokenizer,
    cleaner: EmailCleaner,
    max_length: int,
    num_proc: int,
) -> Dataset:
    def is_valid(example):
        text = example.get("text")
        if text is None or not str(text).strip():
            return False
        label = _extract_label(example)
        if label is None:
            return False
        caption = _extract_caption(example)
        cleaned, _ = cleaner.render(caption, str(text))
        return cleaned is not None

    ds = ds.filter(is_valid, num_proc=num_proc)

    def preprocess(batch):
        texts = batch["text"]
        labels = []
        cleaned_texts = []
        for i, text in enumerate(texts):
            example = {k: v[i] for k, v in batch.items()}
            label = _extract_label(example)
            caption = _extract_caption(example)
            cleaned, _ = cleaner.render(caption, str(text))
            if cleaned is None:
                cleaned = (
                    f"Subject: {cleaner.cfg.default_subject}\n\n"
                    f"{cleaner.cfg.body_prefix}[INVALID SAMPLE]\n"
                )
                label = 0
            labels.append(int(label) if label is not None else 0)
            cleaned_texts.append(cleaned)

        enc = tokenizer(
            cleaned_texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        enc["labels"] = labels
        return enc

    tokenized = ds.map(
        preprocess,
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=True,
        remove_columns=ds.column_names,
    )
    return tokenized


def _prepare_infer_dataset(
    ds: Dataset,
    tokenizer,
    cleaner: EmailCleaner,
    max_length: int,
    num_proc: int,
) -> Tuple[Dataset, Optional[np.ndarray], Dataset]:
    def is_valid(example):
        text = example.get("text")
        return text is not None and str(text).strip()

    ds = ds.filter(is_valid, num_proc=num_proc)

    def preprocess(batch):
        texts = batch["text"]
        cleaned_texts = []
        labels = []
        for i, text in enumerate(texts):
            example = {k: v[i] for k, v in batch.items()}
            label = _extract_label(example)
            caption = _extract_caption(example)
            cleaned, _ = cleaner.render(caption, str(text))
            if cleaned is None:
                cleaned = (
                    f"Subject: {cleaner.cfg.default_subject}\n\n"
                    f"{cleaner.cfg.body_prefix}[INVALID SAMPLE]\n"
                )
                label = 0 if label is None else label
            cleaned_texts.append(cleaned)
            labels.append(-1 if label is None else int(label))

        enc = tokenizer(
            cleaned_texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        enc["labels"] = labels
        return enc

    tokenized = ds.map(
        preprocess,
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=True,
        remove_columns=ds.column_names,
    )

    labels = (
        np.array(tokenized["labels"]) if "labels" in tokenized.column_names else None
    )
    return tokenized, labels, ds


def _resolve_run_dir(output_dir: str) -> str:
    output_path = Path(output_dir)
    if output_path.name == "training_output":
        return str(output_path.parent)
    try:
        hydra_cfg = HydraConfig.get()
        if hydra_cfg.mode == RunMode.RUN:
            return _make_absolute(hydra_cfg.run.dir)
        return _make_absolute(os.path.join(hydra_cfg.sweep.dir, hydra_cfg.sweep.subdir))
    except Exception:
        return str(output_path)


def _save_jsonl(path: Path, records: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _collect_hard_cases(
    source_ds: Dataset,
    logits: np.ndarray,
    labels: Optional[np.ndarray],
    low_conf_threshold: float,
) -> Tuple[List[Dict], List[Dict]]:
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    pred_labels = np.argmax(probs, axis=-1)
    pred_conf = np.max(probs, axis=-1)

    low_conf_cases: List[Dict] = []
    misclassified_cases: List[Dict] = []

    source_data = source_ds.to_list()
    for idx, sample in enumerate(source_data):
        record = {
            "index": idx,
            "pred_label": int(pred_labels[idx]),
            "pred_confidence": float(pred_conf[idx]),
            "phish_probability": float(probs[idx, 1]) if probs.shape[1] > 1 else None,
            "true_label": (
                int(labels[idx]) if labels is not None and labels[idx] >= 0 else None
            ),
            "sample": sample,
        }

        if pred_conf[idx] < low_conf_threshold:
            low_conf_cases.append(record)

        if (
            labels is not None
            and labels[idx] >= 0
            and int(pred_labels[idx]) != int(labels[idx])
        ):
            misclassified_cases.append(record)

    return low_conf_cases, misclassified_cases


def _compute_metrics_from_arrays(
    logits: np.ndarray, labels: np.ndarray
) -> Dict[str, float]:
    preds = np.argmax(logits, axis=-1)
    labels = labels.astype(int)
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }
    try:
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
        metrics["roc_auc"] = roc_auc_score(labels, probs)
    except Exception:
        metrics["roc_auc"] = float("nan")
    return metrics


def _compute_metrics(eval_pred):
    # for HF Trainer
    return _compute_metrics_from_arrays(eval_pred.predictions, eval_pred.label_ids)


def _prediction_summary(logits: np.ndarray) -> Dict[str, float]:
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    return {
        "mean_phish_prob": float(np.mean(probs)),
        "std_phish_prob": float(np.std(probs)),
        "predicted_phish_rate": float(np.mean(probs >= 0.5)),
    }


class WeightedTrainer(Trainer):

    def __init__(self, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch: Optional[int] = None,
        **kwargs,
    ):
        labels = inputs.get("labels")
        outputs = model(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
        )
        logits = outputs.get("logits")
        weights = (
            self.class_weights.to(logits.device)
            if self.class_weights is not None
            else None
        )
        loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def _to_jsonable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class FileMetricsCallback(TrainerCallback):

    def __init__(self, run_dir: str, tensorboard_dir: Optional[str] = None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_jsonl_path = self.run_dir / "intermediate_logs.jsonl"
        self.latest_metrics_path = self.run_dir / "latest_metrics.json"
        self.tensorboard_dir = tensorboard_dir
        self.tb_writer = None

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        if not state.is_world_process_zero or not logs:
            return

        record: Dict = {
            "global_step": int(state.global_step),
            "epoch": float(state.epoch) if state.epoch is not None else None,
        }
        record.update({k: _to_jsonable(v) for k, v in logs.items()})

        with open(self.log_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        with open(self.latest_metrics_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        if self.tensorboard_dir is not None and self.tb_writer is None:
            self.tb_writer = SummaryWriter(log_dir=self.tensorboard_dir)

        if self.tb_writer is not None:
            step = int(state.global_step)
            for key, value in logs.items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    self.tb_writer.add_scalar(key, float(value), step)
            self.tb_writer.flush()

    def on_train_end(self, args, state, control, **kwargs):
        if self.tb_writer is not None:
            self.tb_writer.close()


def _make_absolute(path: str) -> str:
    if os.path.isabs(path):
        return path
    try:
        base_dir = get_original_cwd()
    except Exception:
        base_dir = os.getcwd()
    return os.path.abspath(os.path.join(base_dir, path))


def _build_output_dir(cfg: DictConfig) -> str:
    if getattr(cfg, "training", None) and getattr(cfg.training, "output_dir", None):
        return _make_absolute(str(cfg.training.output_dir))
    try:
        hydra_cfg = HydraConfig.get()
    except Exception:
        return _make_absolute("training_output")

    if hydra_cfg.mode == RunMode.RUN:
        base_dir = hydra_cfg.run.dir
    else:
        base_dir = os.path.join(hydra_cfg.sweep.dir, hydra_cfg.sweep.subdir)

    return _make_absolute(base_dir)


def _resolve_data_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return _make_absolute(path)


def _resolve_logging_dir(
    run_dir: str, configured_logging_dir: Optional[str] = None
) -> str:
    if configured_logging_dir:
        return _make_absolute(str(configured_logging_dir))

    try:
        hydra_cfg = HydraConfig.get()
        if hydra_cfg.mode == RunMode.RUN:
            return str(Path(_make_absolute(hydra_cfg.run.dir)) / "tensorboard")
        return str(
            Path(_make_absolute(hydra_cfg.sweep.dir))
            / "tensorboard"
            / str(hydra_cfg.sweep.subdir)
        )
    except Exception:
        run_path = Path(run_dir)
        return str(run_path / "tensorboard")


def _log_hydra_config(cfg: DictConfig, run_dir: Optional[str]) -> None:
    if not run_dir:
        return
    rank = os.environ.get("RANK", "0")
    if str(rank) != "0":
        return
    cfg_text = OmegaConf.to_yaml(cfg, resolve=True)
    cfg_path = Path(run_dir) / "hydra_config.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(cfg_text)


def _apply_train_ratio(ds: Dataset, ratio: float, seed: int) -> Dataset:
    if ratio >= 1.0:
        return ds
    if ratio <= 0.0:
        raise ValueError(f"train_ratio must be > 0. Got {ratio}")
    total = len(ds)
    keep = max(1, int(total * ratio))
    return ds.shuffle(seed=seed).select(range(keep))


def _run_training(cfg: DictConfig):
    set_seed(cfg.training.seed)

    run_dir = _resolve_run_dir(_build_output_dir(cfg))
    os.makedirs(run_dir, exist_ok=True)

    output_dir = str(Path(run_dir) / "artifacts")
    os.makedirs(output_dir, exist_ok=True)

    final_model_dir = Path(run_dir) / "model"
    final_model_dir.mkdir(parents=True, exist_ok=True)

    cleaner = build_cleaner(cfg.cleaner)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)

    cache_dir = _resolve_data_path(cfg.data.cache_dir)
    extra_train_dir = _resolve_data_path(cfg.data.extra_train_dir)
    augment_data = _resolve_data_path(cfg.data.augment_data)
    generated_eval_dir = _resolve_data_path(cfg.data.generated_eval_dir)

    train_ds = dataset_factory.get_dataset(
        cfg.data.dataset_name, mode="train", cache_dir=cache_dir
    )
    eval_ds = dataset_factory.get_dataset(
        cfg.data.dataset_name, mode="validation", cache_dir=cache_dir
    )

    if extra_train_dir:
        extra_ds = _load_json_data(extra_train_dir)
        train_ds = concatenate_datasets([train_ds, extra_ds])

    if augment_data:
        augment_ds = _load_json_data(augment_data)
        train_ds = concatenate_datasets([train_ds, augment_ds])

    train_ds = _apply_train_ratio(train_ds, cfg.data.train_ratio, cfg.training.seed)

    train_tok = _prepare_dataset(
        train_ds,
        tokenizer,
        cleaner,
        cfg.data.max_length,
        cfg.data.num_proc,
    )
    eval_tok = _prepare_dataset(
        eval_ds,
        tokenizer,
        cleaner,
        cfg.data.max_length,
        cfg.data.num_proc,
    )

    labels = np.array(train_tok["labels"])
    pos = float(np.sum(labels == 1))
    neg = float(np.sum(labels == 0))
    if pos + neg == 0:
        class_weights = None
    else:
        weight_pos = neg / (pos + 1e-6)
        weight_neg = pos / (neg + 1e-6)
        class_weights = torch.tensor([weight_neg, weight_pos], dtype=torch.float)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.name, num_labels=cfg.model.num_labels
    )

    max_steps = getattr(cfg.training, "max_steps", None)
    max_steps = int(max_steps) if max_steps is not None else -1

    report_to = [x for x in list(cfg.training.report_to) if x != "tensorboard"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy=cfg.training.eval_strategy,
        save_strategy=cfg.training.save_strategy,
        learning_rate=cfg.training.learning_rate,
        per_device_train_batch_size=cfg.training.train_batch_size,
        per_device_eval_batch_size=cfg.training.eval_batch_size,
        num_train_epochs=cfg.training.num_train_epochs,
        max_steps=max_steps,
        weight_decay=cfg.training.weight_decay,
        load_best_model_at_end=cfg.training.save_best_model,
        metric_for_best_model=cfg.training.metric_for_best_model,
        greater_is_better=True,
        logging_steps=cfg.training.logging_steps,
        logging_dir=_resolve_logging_dir(
            run_dir, getattr(cfg.training, "logging_dir", None)
        ),
        eval_steps=getattr(cfg.training, "eval_steps", None),
        save_steps=getattr(cfg.training, "save_steps", None),
        save_total_limit=cfg.training.save_total_limit,
        fp16=torch.cuda.is_available(),
        report_to=report_to,
        seed=cfg.training.seed,
    )

    tb_enabled = "tensorboard" in list(cfg.training.report_to)
    if tb_enabled:
        _log_hydra_config(cfg, run_dir)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=_compute_metrics,
        class_weights=class_weights,
    )
    trainer.add_callback(
        FileMetricsCallback(
            run_dir=run_dir,
            tensorboard_dir=training_args.logging_dir if tb_enabled else None,
        )
    )

    trainer.train()
    eval_metrics = trainer.evaluate()

    summary = {
        "train_size": len(train_tok),
        "eval_size": len(eval_tok),
        "model_name": cfg.model.name,
        "cleaner_config": asdict(cleaner.cfg),
        "eval_metrics": {k: float(v) for k, v in eval_metrics.items()},
    }

    hard_case_source_name = "generated_eval" if generated_eval_dir else "validation"
    if generated_eval_dir:
        hard_case_ds = _load_json_data(generated_eval_dir)
    else:
        hard_case_ds = eval_ds

    hard_case_tok, hard_case_labels, hard_case_source_ds = _prepare_infer_dataset(
        hard_case_ds,
        tokenizer,
        cleaner,
        cfg.data.max_length,
        cfg.data.num_proc,
    )
    hard_case_pred = trainer.predict(hard_case_tok)

    if generated_eval_dir:
        pred_summary = _prediction_summary(hard_case_pred.predictions)
        summary["generated_eval"] = {
            "num_samples": len(hard_case_tok),
            "prediction_summary": pred_summary,
        }
        if hard_case_labels is not None and np.any(hard_case_labels >= 0):
            valid_mask = hard_case_labels >= 0
            logits = hard_case_pred.predictions[valid_mask]
            valid_labels = hard_case_labels[valid_mask]
            summary["generated_eval"]["label_metrics"] = _compute_metrics_from_arrays(
                logits, valid_labels
            )

    low_conf_threshold = float(
        getattr(getattr(cfg, "hard_case", {}), "low_conf_threshold", 0.60)
    )
    low_conf_cases, misclassified_cases = _collect_hard_cases(
        hard_case_source_ds,
        hard_case_pred.predictions,
        hard_case_labels,
        low_conf_threshold,
    )

    if trainer.is_world_process_zero():
        hard_case_dir = Path(_resolve_run_dir(output_dir)) / "hard_case"
        hard_case_dir.mkdir(parents=True, exist_ok=True)

        low_conf_path = hard_case_dir / "low_confidence.jsonl"
        misclassified_path = hard_case_dir / "misclassified.jsonl"

        _save_jsonl(low_conf_path, low_conf_cases)
        _save_jsonl(misclassified_path, misclassified_cases)

        summary["hard_case"] = {
            "source": hard_case_source_name,
            "dir": str(hard_case_dir),
            "low_conf_threshold": low_conf_threshold,
            "num_samples": len(hard_case_tok),
            "low_confidence_count": len(low_conf_cases),
            "misclassified_count": len(misclassified_cases),
            "low_confidence_file": str(low_conf_path),
            "misclassified_file": str(misclassified_path),
        }

        if generated_eval_dir and "generated_eval" in summary:
            summary["generated_eval"]["hard_case"] = dict(summary["hard_case"])

    summary["intermediate_log_files"] = {
        "jsonl": str(Path(run_dir) / "intermediate_logs.jsonl"),
        "latest": str(Path(run_dir) / "latest_metrics.json"),
    }

    if trainer.is_world_process_zero():
        with open(Path(run_dir) / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        trainer.save_model(str(final_model_dir))
        tokenizer.save_pretrained(str(final_model_dir))


def _distributed_worker(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    cfg_container: Dict,
):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    cfg = OmegaConf.create(cfg_container)
    _run_training(cfg)


@hydra.main(version_base=None, config_path=".", config_name="config_detect")
def main(cfg: DictConfig):
    if cfg.training.ngpus > 1 and "LOCAL_RANK" not in os.environ:
        cfg_container = OmegaConf.to_container(cfg, resolve=True)
        training_cfg = cfg_container.get("training", {})
        try:
            hydra_cfg = HydraConfig.get()
            if hydra_cfg.mode == RunMode.RUN:
                runtime_output_dir = _make_absolute(hydra_cfg.run.dir)
                runtime_logging_dir = str(Path(runtime_output_dir) / "tensorboard")
            else:
                runtime_output_dir = _make_absolute(
                    os.path.join(hydra_cfg.sweep.dir, hydra_cfg.sweep.subdir)
                )
                runtime_logging_dir = str(
                    Path(_make_absolute(hydra_cfg.sweep.dir))
                    / "tensorboard"
                    / str(hydra_cfg.sweep.subdir)
                )
        except Exception:
            runtime_output_dir = None
            runtime_logging_dir = None

        if runtime_output_dir is not None:
            training_cfg["output_dir"] = runtime_output_dir
            training_cfg["logging_dir"] = runtime_logging_dir
        elif training_cfg.get("output_dir"):
            training_cfg["output_dir"] = _make_absolute(training_cfg["output_dir"])
        data_cfg = cfg_container.get("data", {})
        for key in (
            "cache_dir",
            "extra_train_dir",
            "augment_data",
            "generated_eval_dir",
        ):
            if data_cfg.get(key):
                data_cfg[key] = _make_absolute(data_cfg[key])
        mp.spawn(
            _distributed_worker,
            args=(
                cfg.training.ngpus,
                cfg.training.master_addr,
                cfg.training.master_port,
                cfg_container,
            ),
            nprocs=cfg.training.ngpus,
            join=True,
        )
        return

    _run_training(cfg)


if __name__ == "__main__":
    main()
