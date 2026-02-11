import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

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


def _load_json_dir(json_dir: str) -> Dataset:
    data_root = Path(json_dir)
    if not data_root.exists():
        raise FileNotFoundError(f"Data directory not found: {data_root}")
    json_files = sorted(data_root.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {data_root}")
    ds = load_dataset("json", data_files=[str(p) for p in json_files])["train"]
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
) -> Tuple[Dataset, Optional[np.ndarray]]:
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
    return tokenized, labels


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


def _resolve_logging_dir(cfg: DictConfig, output_dir: str) -> str:
    logging_dir = getattr(cfg.training, "logging_dir", None)
    if logging_dir:
        return _make_absolute(str(logging_dir))
    return output_dir


def _log_hydra_config(cfg: DictConfig, log_dir: Optional[str]) -> None:
    if not log_dir:
        return
    rank = os.environ.get("RANK", "0")
    if str(rank) != "0":
        return
    cfg_text = OmegaConf.to_yaml(cfg, resolve=True)
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text("hydra_cfg", f"```yaml\n{cfg_text}\n```", 0)
    writer.flush()
    writer.close()


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

    output_dir = _build_output_dir(cfg)
    os.makedirs(output_dir, exist_ok=True)

    cleaner = build_cleaner(cfg.cleaner)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)

    cache_dir = _resolve_data_path(cfg.data.cache_dir)
    extra_train_dir = _resolve_data_path(cfg.data.extra_train_dir)
    generated_eval_dir = _resolve_data_path(cfg.data.generated_eval_dir)

    train_ds = dataset_factory.get_dataset(
        cfg.data.dataset_name, mode="train", cache_dir=cache_dir
    )
    eval_ds = dataset_factory.get_dataset(
        cfg.data.dataset_name, mode="validation", cache_dir=cache_dir
    )

    if extra_train_dir:
        extra_ds = _load_json_dir(extra_train_dir)
        train_ds = concatenate_datasets([train_ds, extra_ds])

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
        logging_dir=_resolve_logging_dir(cfg, output_dir),
        eval_steps=getattr(cfg.training, "eval_steps", None),
        save_steps=getattr(cfg.training, "save_steps", None),
        save_total_limit=cfg.training.save_total_limit,
        fp16=torch.cuda.is_available(),
        report_to=list(cfg.training.report_to),
        seed=cfg.training.seed,
    )

    if "tensorboard" in list(cfg.training.report_to):
        _log_hydra_config(cfg, training_args.logging_dir)

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

    trainer.train()
    eval_metrics = trainer.evaluate()

    summary = {
        "train_size": len(train_tok),
        "eval_size": len(eval_tok),
        "model_name": cfg.model.name,
        "cleaner_config": asdict(cleaner.cfg),
        "eval_metrics": {k: float(v) for k, v in eval_metrics.items()},
    }

    if generated_eval_dir:
        gen_ds = _load_json_dir(generated_eval_dir)
        gen_tok, labels = _prepare_infer_dataset(
            gen_ds,
            tokenizer,
            cleaner,
            cfg.data.max_length,
            cfg.data.num_proc,
        )
        pred = trainer.predict(gen_tok)
        pred_summary = _prediction_summary(pred.predictions)
        summary["generated_eval"] = {
            "num_samples": len(gen_tok),
            "prediction_summary": pred_summary,
        }
        if labels is not None and np.any(labels >= 0):
            valid_mask = labels >= 0
            logits = pred.predictions[valid_mask]
            valid_labels = labels[valid_mask]
            summary["generated_eval"]["label_metrics"] = _compute_metrics_from_arrays(
                logits, valid_labels
            )

    if trainer.is_world_process_zero():
        with open(Path(output_dir) / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)


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
        if training_cfg.get("output_dir"):
            training_cfg["output_dir"] = _make_absolute(training_cfg["output_dir"])
        data_cfg = cfg_container.get("data", {})
        for key in ("cache_dir", "extra_train_dir", "generated_eval_dir"):
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
