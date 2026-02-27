import argparse
import concurrent.futures
import json
import logging
import random
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm


def _setup_import_path() -> Path:
    current = Path(__file__).resolve()
    repo_root = None
    for parent in current.parents:
        if (parent / "src").exists():
            repo_root = parent
            break
    if repo_root is None:
        raise RuntimeError("Cannot find repository root containing src/")
    sys.path.insert(0, str(repo_root / "src"))
    return repo_root


REPO_ROOT = _setup_import_path()

import sample.sampling as sampling  # noqa: E402
from sample.load_model import load_model  # noqa: E402
from utils import utils  # noqa: E402
from utils.tokenizer_factory import get_caption_tokenizer, get_text_tokenizer  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def truncate_at_eos(batch_ids, eos_id):
    output = []
    for row in batch_ids.tolist():
        if eos_id in row:
            idx = row.index(eos_id)
            output.append(row[:idx])
        else:
            output.append(row)
    return output


def clean_generated_sample(text: str) -> str:
    if not text:
        return text
    body_match = re.search(r"body:\s*", text, flags=re.IGNORECASE)
    if body_match:
        text = text[body_match.end() :]
    pattern = re.compile(
        r"\bThis is a\s+(?:benign|phish)\s+email\.\s*",
        re.IGNORECASE,
    )
    cleaned = pattern.sub("", text)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def parse_gpu_ids(gpu_ids_str: str) -> list[int]:
    gpu_ids_str = (gpu_ids_str or "").strip()
    if gpu_ids_str == "":
        return []
    gpu_ids = []
    for token in gpu_ids_str.split(","):
        token = token.strip()
        if token == "":
            continue
        gpu_id = int(token)
        if gpu_id < 0:
            raise ValueError("GPU id must be >= 0")
        gpu_ids.append(gpu_id)
    if len(gpu_ids) == 0:
        raise ValueError("--gpu_ids is provided but empty")
    return gpu_ids


def split_records(records: list[dict], shards: int) -> list[list[dict]]:
    outputs = [[] for _ in range(shards)]
    for idx, record in enumerate(records):
        outputs[idx % shards].append(record)
    return outputs


def load_captions(caption_file: Path) -> list[str]:
    if not caption_file.exists():
        raise FileNotFoundError(f"Caption file not found: {caption_file}")
    if not caption_file.is_file():
        raise ValueError(f"Caption file is not a file: {caption_file}")

    captions = []
    with open(caption_file, encoding="utf-8") as f:
        for line in f:
            caption = line.strip()
            if caption:
                captions.append(caption)
    if len(captions) == 0:
        raise ValueError(f"No non-empty captions found in: {caption_file}")
    return captions


def _generate_on_device(
    device_str: str,
    worker_id: int,
    worker_seed: int,
    batch_records: list[dict],
    generation_cfg: dict,
) -> list[dict]:
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    device = torch.device(device_str)
    model, graph, noise = load_model(generation_cfg["model_path"], device)

    cfg = None
    hydra_cfg = Path(generation_cfg["model_path"]) / ".hydra" / "config.yaml"
    if hydra_cfg.exists():
        cfg = utils.load_hydra_config_from_run(generation_cfg["model_path"])
    model_cfg = getattr(model, "config", None)
    cfg = cfg or model_cfg

    length = generation_cfg["length"]
    steps = generation_cfg["steps"]
    predictor = generation_cfg["predictor"]
    denoise = generation_cfg["denoise"]
    caption_max_length = generation_cfg["caption_max_length"]
    cfg_scale = generation_cfg["cfg_scale"]
    mask_mode = generation_cfg["mask_mode"]

    if length is None and cfg is not None:
        length = getattr(getattr(cfg, "model", None), "length", None)
    length = length or 1024

    if steps is None and cfg is not None:
        steps = getattr(getattr(cfg, "sampling", None), "steps", None)
    steps = steps or 128

    if cfg is not None:
        predictor = getattr(getattr(cfg, "sampling", None), "predictor", predictor)
        denoise = getattr(getattr(cfg, "sampling", None), "noise_removal", denoise)

    text_tokenizer_name = generation_cfg["text_tokenizer_name"]
    caption_tokenizer_name = generation_cfg["caption_tokenizer_name"]
    if cfg is not None:
        text_tokenizer_name = getattr(
            getattr(cfg, "tokenizer", None), "text", text_tokenizer_name
        )
        caption_tokenizer_name = getattr(
            getattr(cfg, "tokenizer", None), "caption", caption_tokenizer_name
        )
        caption_max_length = int(
            getattr(
                getattr(cfg, "data", None), "caption_max_length", caption_max_length
            )
        )

    tokenizer_text = get_text_tokenizer(text_tokenizer_name)
    tokenizer_caption = get_caption_tokenizer(caption_tokenizer_name)

    generated_output = []
    batch_size = generation_cfg["batch_size"]
    total_batches = (len(batch_records) + batch_size - 1) // batch_size

    with torch.inference_mode():
        for start in tqdm(
            range(0, len(batch_records), batch_size),
            total=total_batches,
            desc=f"Worker-{worker_id} generating",
            position=worker_id,
            leave=True,
        ):
            records = batch_records[start : start + batch_size]

            captions = [r["style_caption"] for r in records]
            source_texts = [r.get("source_text") or r["style_caption"] for r in records]
            labels = [int(r["label"]) for r in records]

            enc_cap = tokenizer_caption(
                captions,
                return_attention_mask=True,
                add_special_tokens=True,
                max_length=caption_max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            if mask_mode == "full":
                eval_text_mask = torch.ones(
                    (len(records), length), dtype=torch.int32, device=device
                )
            else:
                enc_text = tokenizer_text(
                    source_texts,
                    return_attention_mask=True,
                    add_special_tokens=True,
                    max_length=length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                eval_text_mask = enc_text["attention_mask"].to(device)
            eval_style_caption = enc_cap["input_ids"].to(device)
            eval_style_caption_mask = enc_cap["attention_mask"].to(device)

            sampling_fn = sampling.PCSampler(
                graph,
                noise,
                (len(records), length),
                predictor,
                steps,
                denoise,
                1e-5,
                device,
            )

            samples = sampling_fn(
                model,
                eval_text_mask,
                eval_style_caption,
                eval_style_caption_mask,
                cfg_scale,
            )

            sample_trunc = truncate_at_eos(samples, tokenizer_text.eos_token_id)
            sentences = tokenizer_text.batch_decode(sample_trunc)

            for sentence, record, label in zip(sentences, records, labels):
                generated_output.append(
                    {
                        "sample": clean_generated_sample(sentence),
                        "label": int(label),
                        "style_caption": record["style_caption"],
                        "source_text": record["source_text"],
                    }
                )

    logger.info(
        "Worker-%d finished on %s, generated %d samples",
        worker_id,
        device_str,
        len(generated_output),
    )
    return generated_output


def main():
    parser = argparse.ArgumentParser(
        description="Generate augmented samples from one caption or caption file"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="exp_local/phish-email/2026.02.02/105439/",
    )
    parser.add_argument("--caption", type=str, default="", help="Manual input caption")
    parser.add_argument(
        "--caption_file",
        type=str,
        default="src/detection_model/caption.txt",
        help="Path to caption file, one caption per line.",
    )
    parser.add_argument(
        "--source_text",
        type=str,
        default="",
        help="Optional source text used to build attention mask; default empty text.",
    )
    parser.add_argument("--label", type=int, default=1)
    parser.add_argument("--num_generate", type=int, default=-1)
    parser.add_argument(
        "--per_caption_generate",
        type=int,
        default=3,
        help="Used when --num_generate < 0: generate this many samples per caption.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="src/detection_model/augmented_from_caption.jsonl",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument(
        "--mask_mode",
        type=str,
        default="full",
        choices=["full", "source"],
        help="Generation text mask mode: full=all ones, source=from source_text tokenizer attention mask.",
    )
    parser.add_argument("--cfg_scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="0,1",
        help="Comma-separated GPU ids, e.g. '0,1,2'. Empty means all visible CUDA devices.",
    )
    args = parser.parse_args()

    caption_text = args.caption.strip()
    caption_file_text = (args.caption_file or "").strip()
    if not caption_text and not caption_file_text:
        raise ValueError("Provide either --caption or --caption_file")
    if caption_text and caption_file_text:
        raise ValueError("--caption and --caption_file are mutually exclusive")
    if args.num_generate == 0:
        raise ValueError(
            "--num_generate must be > 0, or < 0 to enable per-caption mode"
        )
    if args.per_caption_generate <= 0:
        raise ValueError("--per_caption_generate must be > 0")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_path = Path(args.model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    generation_cfg = {
        "model_path": str(model_path),
        "length": args.length,
        "mask_mode": args.mask_mode,
        "steps": args.steps,
        "predictor": "analytic",
        "denoise": True,
        "text_tokenizer_name": "gpt2",
        "caption_tokenizer_name": "bert-base-uncased",
        "caption_max_length": 256,
        "cfg_scale": args.cfg_scale,
        "batch_size": args.batch_size,
    }

    if caption_file_text:
        captions = load_captions(Path(caption_file_text).resolve())
    else:
        captions = [caption_text]

    random.shuffle(captions)
    generation_records = []
    base_source_text = (args.source_text or "").strip()
    if args.num_generate < 0:
        for caption in captions:
            source_text = base_source_text if base_source_text else caption
            for _ in range(args.per_caption_generate):
                generation_records.append(
                    {
                        "style_caption": caption,
                        "source_text": source_text,
                        "label": int(args.label),
                    }
                )
        generation_mode = f"per_caption({args.per_caption_generate})"
    else:
        base_count = args.num_generate // len(captions)
        remainder = args.num_generate % len(captions)
        for idx, caption in enumerate(captions):
            repeat_n = base_count + (1 if idx < remainder else 0)
            source_text = base_source_text if base_source_text else caption
            for _ in range(repeat_n):
                generation_records.append(
                    {
                        "style_caption": caption,
                        "source_text": source_text,
                        "label": int(args.label),
                    }
                )
        generation_mode = f"total({args.num_generate})"
    random.shuffle(generation_records)
    logger.info(
        "Loaded %d caption(s), target generate=%d, mode=%s, mask_mode=%s",
        len(captions),
        len(generation_records),
        generation_mode,
        args.mask_mode,
    )
    if args.mask_mode == "source" and not base_source_text:
        logger.warning(
            "--source_text is empty in source mask mode; fallback to caption text as source_text."
        )

    generated_output = []
    if torch.cuda.is_available():
        available = torch.cuda.device_count()
        requested_gpu_ids = parse_gpu_ids(args.gpu_ids)
        if len(requested_gpu_ids) == 0:
            gpu_ids = list(range(available))
        else:
            for gpu_id in requested_gpu_ids:
                if gpu_id >= available:
                    raise ValueError(
                        f"Requested GPU id {gpu_id} out of range. Available count={available}."
                    )
            gpu_ids = requested_gpu_ids

        if len(gpu_ids) <= 1:
            device_str = f"cuda:{gpu_ids[0]}" if len(gpu_ids) == 1 else "cuda:0"
            logger.info("Using single GPU: %s", device_str)
            generated_output = _generate_on_device(
                device_str,
                0,
                args.seed,
                generation_records,
                generation_cfg,
            )
        else:
            logger.info("Using multi-GPU: %s", gpu_ids)
            shards = split_records(generation_records, len(gpu_ids))
            for idx, shard in enumerate(shards):
                logger.info("shard-%d size: %d", idx, len(shard))

            ctx = torch.multiprocessing.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=len(gpu_ids), mp_context=ctx
            ) as executor:
                futures = []
                for worker_id, (gpu_id, shard_records) in enumerate(
                    zip(gpu_ids, shards)
                ):
                    if len(shard_records) == 0:
                        continue
                    futures.append(
                        executor.submit(
                            _generate_on_device,
                            f"cuda:{gpu_id}",
                            worker_id,
                            args.seed + worker_id,
                            shard_records,
                            generation_cfg,
                        )
                    )

                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Waiting workers",
                ):
                    generated_output.extend(future.result())
    else:
        logger.info("CUDA not available, fallback to CPU")
        generated_output = _generate_on_device(
            "cpu",
            0,
            args.seed,
            generation_records,
            generation_cfg,
        )

    if args.output_path:
        output_path = Path(args.output_path).resolve()
    else:
        output_dir = Path.cwd() / "temp"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "augmented_from_caption.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out_f:
        for record in generated_output:
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Saved %d samples to: %s", len(generated_output), output_path)


if __name__ == "__main__":
    main()
