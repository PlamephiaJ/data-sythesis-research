import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import hydra
import numpy as np
import torch
import torch.multiprocessing as mp
from datasets import Dataset, concatenate_datasets
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    set_seed,
)

from data_process import dataset_factory


try:
    from detection_model import train as base_train
except Exception:
    import train as base_train


logger = logging.getLogger(__name__)


def _with_required_columns(ds: Dataset) -> Dataset:
    required_cols = [
        "text",
        "phish",
        "style_caption",
        "caption",
        "subject",
        "cluster_id",
    ]

    def _project_required(example: Dict) -> Dict:
        return {col: example.get(col, None) for col in required_cols}

    return ds.map(_project_required, remove_columns=ds.column_names)


def _normalize_augment_dataset(ds: Dataset) -> Dataset:
    if "text" not in ds.column_names:
        if "sample" in ds.column_names:
            ds = ds.rename_column("sample", "text")
        else:
            raise ValueError("augment_data must contain `text` or `sample` column.")

    if "phish" not in ds.column_names:
        if "label" in ds.column_names:
            ds = ds.rename_column("label", "phish")
        else:
            raise ValueError("augment_data must contain `phish` or `label` column.")

    def _coerce_fields(example: Dict):
        text = example.get("text")
        example["text"] = "" if text is None else str(text)
        phish = example.get("phish")
        example["phish"] = int(phish) if phish is not None else 0
        return example

    ds = ds.map(_coerce_fields)
    return _with_required_columns(ds)


def _parse_cluster_id(value) -> Optional[int]:
    if value is None:
        return None
    try:
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                return None
            if "." in value:
                return int(float(value))
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_int_label(value, default: int = 0) -> int:
    if value is None:
        return default
    try:
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                return default
            if "." in value:
                return int(float(value))
        return int(value)
    except (TypeError, ValueError):
        return default


def _sample_to_size(ds: Dataset, target_size: int, seed: int) -> Dataset:
    if target_size <= 0 or len(ds) == 0:
        return ds.select([])
    if len(ds) <= target_size:
        return ds
    return ds.train_test_split(train_size=target_size, seed=seed)["train"]


def _build_aug_train_eval_dataset(
    cfg: DictConfig,
) -> Tuple[Dataset, Dataset]:
    augment_data = getattr(
        cfg.data,
        "augment_data",
        "src/detection_model/augmented_from_caption_all.jsonl",
    )
    augment_data = base_train._resolve_data_path(augment_data)
    if not augment_data:
        raise ValueError("data.augment_data is empty or invalid.")

    train_ds = _normalize_augment_dataset(base_train._load_json_data(augment_data))
    cache_dir = base_train._resolve_data_path(cfg.data.cache_dir)
    base_train_ds = dataset_factory.get_dataset(
        cfg.data.dataset_name, mode="train", cache_dir=cache_dir
    )
    base_eval_ds = dataset_factory.get_dataset(
        cfg.data.dataset_name, mode="validation", cache_dir=cache_dir
    )
    split_seed = int(getattr(cfg.data, "split_seed", 42))

    if "cluster_id" in base_eval_ds.column_names:
        eval_cluster_ds = base_eval_ds.filter(
            lambda x: _parse_cluster_id(x.get("cluster_id")) == 0,
            num_proc=cfg.data.num_proc,
        )
    else:
        if "id" not in base_eval_ds.column_names:
            raise ValueError(
                "Validation dataset has neither `cluster_id` nor `id`, cannot build cluster_id==0 eval set."
            )

        masked_phish_data = getattr(
            cfg.data,
            "masked_phish_data",
            "data_phish/masked/phish_clustered_all.jsonl",
        )
        masked_phish_data = base_train._resolve_data_path(masked_phish_data)
        masked_all_ds = base_train._load_json_data(masked_phish_data)
        if (
            "id" not in masked_all_ds.column_names
            or "cluster_id" not in masked_all_ds.column_names
        ):
            raise ValueError(
                "masked_phish_data must contain both `id` and `cluster_id` for eval mapping."
            )

        cluster0_ids = set()
        for sample_id, cluster_id in zip(
            masked_all_ds["id"], masked_all_ds["cluster_id"]
        ):
            if _parse_cluster_id(cluster_id) == 0:
                cluster0_ids.add(int(sample_id))

        eval_cluster_ds = base_eval_ds.filter(
            lambda x: int(x.get("id")) in cluster0_ids,
            num_proc=cfg.data.num_proc,
        )

    eval_cluster_count = len(eval_cluster_ds)

    train_benign_source = base_train_ds.filter(
        lambda x: _to_int_label(x.get("phish"), default=0) == 0,
        num_proc=cfg.data.num_proc,
    )
    train_benign_ds = _sample_to_size(
        train_benign_source,
        target_size=len(train_ds),
        seed=split_seed,
    )

    if len(train_benign_ds) < len(train_ds):
        logger.warning(
            "Train benign pool is smaller than augment set: benign=%s, augment=%s. Using all available benign.",
            len(train_benign_ds),
            len(train_ds),
        )

    train_ds = concatenate_datasets(
        [
            train_ds,
            _with_required_columns(train_benign_ds),
        ]
    )

    eval_benign_source = base_eval_ds.filter(
        lambda x: _to_int_label(x.get("phish"), default=0) == 0,
        num_proc=cfg.data.num_proc,
    )

    if "id" in eval_cluster_ds.column_names and "id" in eval_benign_source.column_names:
        eval_cluster_ids = {int(v) for v in eval_cluster_ds["id"]}
        eval_benign_source = eval_benign_source.filter(
            lambda x: int(x.get("id")) not in eval_cluster_ids,
            num_proc=cfg.data.num_proc,
        )

    eval_benign_ds = _sample_to_size(
        eval_benign_source,
        target_size=eval_cluster_count,
        seed=split_seed,
    )

    if len(eval_benign_ds) < eval_cluster_count:
        logger.warning(
            "Eval benign pool is smaller than cluster-0 set: benign=%s, cluster0=%s. Using all available benign.",
            len(eval_benign_ds),
            eval_cluster_count,
        )

    eval_ds = concatenate_datasets(
        [
            _with_required_columns(eval_cluster_ds),
            _with_required_columns(eval_benign_ds),
        ]
    )

    if len(eval_ds) == 0:
        raise ValueError(
            "Validation set is empty after building cluster_id==0 + benign eval set."
        )

    logger.info(
        "Prepared datasets for train_with_augment: train=%s (augment + benign=%s), eval=%s (cluster0=%s + benign=%s)",
        len(train_ds),
        len(train_benign_ds),
        len(eval_ds),
        eval_cluster_count,
        len(eval_benign_ds),
    )

    return train_ds, eval_ds


def _run_training(cfg: DictConfig):
    set_seed(cfg.training.seed)

    run_dir = base_train._resolve_run_dir(base_train._build_output_dir(cfg))
    os.makedirs(run_dir, exist_ok=True)
    log_file = base_train._setup_logging(run_dir)

    try:
        logger.info(
            "Starting training with augment-only train set. run_dir=%s", run_dir
        )

        output_dir = str(Path(run_dir) / "artifacts")
        os.makedirs(output_dir, exist_ok=True)

        final_model_dir = Path(run_dir) / "model"
        final_model_dir.mkdir(parents=True, exist_ok=True)

        cleaner = base_train.build_cleaner(cfg.cleaner)
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)

        used_max_length = base_train._effective_max_length(
            cfg.data.max_length, tokenizer, cfg.model.name
        )
        if int(cfg.data.max_length) != int(used_max_length):
            logger.warning(
                "Requested data.max_length=%s but model/tokenizer supports at most %s; clamping.",
                int(cfg.data.max_length),
                int(used_max_length),
            )

        generated_eval_dir = base_train._resolve_data_path(cfg.data.generated_eval_dir)

        train_ds, eval_ds = _build_aug_train_eval_dataset(cfg)

        train_ds = base_train._apply_train_ratio(
            train_ds, cfg.data.train_ratio, cfg.training.seed
        )

        logger.info("Final training dataset size: %s samples.", len(train_ds))
        logger.info("Final validation dataset size: %s samples.", len(eval_ds))

        train_tok = base_train._prepare_dataset(
            train_ds,
            tokenizer,
            cleaner,
            used_max_length,
            cfg.data.num_proc,
        )
        eval_tok = base_train._prepare_dataset(
            eval_ds,
            tokenizer,
            cleaner,
            used_max_length,
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

        save_best_model = bool(getattr(cfg.training, "save_best_model", False))
        save_strategy = "best" if save_best_model else cfg.training.save_strategy
        save_total_limit = 1 if save_best_model else cfg.training.save_total_limit
        save_steps = (
            getattr(cfg.training, "save_steps", None)
            if str(save_strategy) == "steps"
            else None
        )

        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy=cfg.training.eval_strategy,
            save_strategy=save_strategy,
            learning_rate=cfg.training.learning_rate,
            per_device_train_batch_size=cfg.training.train_batch_size,
            per_device_eval_batch_size=cfg.training.eval_batch_size,
            num_train_epochs=cfg.training.num_train_epochs,
            max_steps=max_steps,
            weight_decay=cfg.training.weight_decay,
            load_best_model_at_end=save_best_model,
            metric_for_best_model=cfg.training.metric_for_best_model,
            greater_is_better=True,
            logging_steps=cfg.training.logging_steps,
            logging_dir=base_train._resolve_logging_dir(
                run_dir, getattr(cfg.training, "logging_dir", None)
            ),
            eval_steps=getattr(cfg.training, "eval_steps", None),
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            fp16=torch.cuda.is_available(),
            report_to=report_to,
            seed=cfg.training.seed,
        )

        tb_enabled = "tensorboard" in list(cfg.training.report_to)
        if tb_enabled:
            base_train._log_hydra_config(cfg, run_dir)

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        trainer = base_train.WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_tok,
            eval_dataset=eval_tok,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=base_train._compute_metrics,
            class_weights=class_weights,
        )
        trainer.add_callback(
            base_train.FileMetricsCallback(
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
            "train_data": {
                "augment_source": getattr(
                    cfg.data,
                    "augment_data",
                    "src/detection_model/augmented_from_caption_all.jsonl",
                ),
                "benign_source": cfg.data.dataset_name,
                "benign_split": "train",
                "benign_target": "same_as_augment",
            },
            "eval_data": {
                "source": cfg.data.dataset_name,
                "split": "validation",
                "cluster_id": 0,
                "benign_target": "same_as_cluster0",
            },
        }

        if generated_eval_dir:
            hard_case_ds = base_train._load_json_data(generated_eval_dir)
        else:
            hard_case_ds = eval_ds

        hard_case_tok, hard_case_labels, _ = base_train._prepare_infer_dataset(
            _with_required_columns(hard_case_ds),
            tokenizer,
            cleaner,
            used_max_length,
            cfg.data.num_proc,
        )
        hard_case_pred = trainer.predict(hard_case_tok)

        if generated_eval_dir:
            pred_summary = base_train._prediction_summary(hard_case_pred.predictions)
            summary["generated_eval"] = {
                "num_samples": len(hard_case_tok),
                "prediction_summary": pred_summary,
            }
            if hard_case_labels is not None and np.any(hard_case_labels >= 0):
                valid_mask = hard_case_labels >= 0
                logits = hard_case_pred.predictions[valid_mask]
                valid_labels = hard_case_labels[valid_mask]
                summary["generated_eval"]["label_metrics"] = (
                    base_train._compute_metrics_from_arrays(logits, valid_labels)
                )

        summary["intermediate_log_files"] = {
            "jsonl": str(Path(run_dir) / "intermediate_logs.jsonl"),
            "latest": str(Path(run_dir) / "latest_metrics.json"),
        }

        if trainer.is_world_process_zero():
            with open(Path(run_dir) / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            trainer.save_model(str(final_model_dir))
            tokenizer.save_pretrained(str(final_model_dir))

        logger.info("Training with augment-only train set finished successfully.")

    except Exception:
        logger.exception("Training crashed. See log file: %s", log_file)
        raise


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
        device_count = torch.cuda.device_count()
        if rank >= device_count:
            raise RuntimeError(
                f"LOCAL_RANK={rank} is out of range for visible CUDA devices ({device_count})."
            )
        torch.cuda.set_device(rank)
    cfg = OmegaConf.create(cfg_container)
    _run_training(cfg)


@hydra.main(version_base=None, config_path=".", config_name="config_detect")
def main(cfg: DictConfig):
    requested_ngpus = int(cfg.training.ngpus)
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if available_gpus <= 0:
        effective_ngpus = 1
    elif requested_ngpus > available_gpus:
        effective_ngpus = available_gpus
    else:
        effective_ngpus = requested_ngpus

    if effective_ngpus != requested_ngpus:
        logger.warning(
            "Requested training.ngpus=%s but only %s CUDA device(s) are available. Using ngpus=%s.",
            requested_ngpus,
            available_gpus,
            effective_ngpus,
        )

    cfg.training.ngpus = effective_ngpus

    if cfg.training.ngpus > 1 and "LOCAL_RANK" not in os.environ:
        cfg_container = OmegaConf.to_container(cfg, resolve=True)
        training_cfg = cfg_container.get("training", {})
        training_cfg["ngpus"] = effective_ngpus
        try:
            hydra_cfg = HydraConfig.get()
            if hydra_cfg.mode == RunMode.RUN:
                runtime_output_dir = base_train._make_absolute(hydra_cfg.run.dir)
                runtime_logging_dir = str(Path(runtime_output_dir) / "tensorboard")
            else:
                runtime_output_dir = base_train._make_absolute(
                    os.path.join(hydra_cfg.sweep.dir, hydra_cfg.sweep.subdir)
                )
                runtime_logging_dir = str(
                    Path(base_train._make_absolute(hydra_cfg.sweep.dir))
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
            training_cfg["output_dir"] = base_train._make_absolute(
                training_cfg["output_dir"]
            )
        data_cfg = cfg_container.get("data", {})
        for key in (
            "cache_dir",
            "extra_train_dir",
            "augment_data",
            "generated_eval_dir",
            "masked_phish_data",
        ):
            if data_cfg.get(key):
                data_cfg[key] = base_train._make_absolute(data_cfg[key])
        mp.spawn(
            _distributed_worker,
            args=(
                effective_ngpus,
                cfg.training.master_addr,
                cfg.training.master_port,
                cfg_container,
            ),
            nprocs=effective_ngpus,
            join=True,
        )
        return

    _run_training(cfg)


if __name__ == "__main__":
    main()
