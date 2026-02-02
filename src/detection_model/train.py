import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
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


def build_cleaner() -> EmailCleaner:
    return EmailCleaner(
        EmailCleanConfig(
            render_clean_email=True,
            mask_urls=True,
            mask_emails=True,
            mask_phones=True,
            truncate_on_thread_markers=True,
            truncate_on_long_quote_block=True,
            strip_common_disclaimers=True,
            drop_if_symbol_ratio_gt=0.60,
            max_body_chars=4000,
        )
    )


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


def _compute_metrics(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
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

    def compute_loss(self, model, inputs, return_outputs=False):
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


def _build_output_dir(output_dir: Optional[str]) -> str:
    if output_dir:
        return output_dir
    timestamp = datetime.now().strftime("%Y.%m.%d/%H%M%S")
    return str(Path("exp_local") / "detection_model" / timestamp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--cache_dir", type=str, default="data_phish/json")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--num_proc", type=int, default=120)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extra_train_dir", type=str, default=None)
    parser.add_argument("--generated_eval_dir", type=str, default=None)
    parser.add_argument("--save_best_model", action="store_true", default=True)
    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = _build_output_dir(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    cleaner = build_cleaner()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_ds = dataset_factory.get_dataset(
        "phish-email", mode="train", cache_dir=args.cache_dir
    )
    eval_ds = dataset_factory.get_dataset(
        "phish-email", mode="validation", cache_dir=args.cache_dir
    )

    if args.extra_train_dir:
        extra_ds = _load_json_dir(args.extra_train_dir)
        train_ds = concatenate_datasets([train_ds, extra_ds])

    train_tok = _prepare_dataset(
        train_ds, tokenizer, cleaner, args.max_length, args.num_proc
    )
    eval_tok = _prepare_dataset(
        eval_ds, tokenizer, cleaner, args.max_length, args.num_proc
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
        args.model_name, num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        load_best_model_at_end=args.save_best_model,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to=["tensorboard"],
        seed=args.seed,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=_compute_metrics,
        class_weights=class_weights,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()

    summary = {
        "train_size": len(train_tok),
        "eval_size": len(eval_tok),
        "model_name": args.model_name,
        "cleaner_config": asdict(cleaner.cfg),
        "eval_metrics": {k: float(v) for k, v in eval_metrics.items()},
    }

    if args.generated_eval_dir:
        gen_ds = _load_json_dir(args.generated_eval_dir)
        gen_tok, labels = _prepare_infer_dataset(
            gen_ds, tokenizer, cleaner, args.max_length, args.num_proc
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
            summary["generated_eval"]["label_metrics"] = _compute_metrics(
                (logits, valid_labels)
            )

    with open(Path(output_dir) / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
