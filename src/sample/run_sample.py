import argparse
import json
import logging
import math
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import sample.sampling as sampling
from data_process import data as data_process
from sample.load_model import load_model
from utils import utils
from utils.eval_factory import get_alignment_metric, get_eval_lm, get_mauve_score
from utils.tokenizer_factory import get_caption_tokenizer, get_text_tokenizer


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)

    return distributed, rank, world_size, local_rank


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)
    distributed, rank, world_size, local_rank = setup_distributed()

    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument(
        "--model_path", default="exp_local/phish-email/2026.02.02/105439/", type=str
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--length", type=int, default=None)
    parser.add_argument("--cfg_scale", type=float, default=0.0)
    args = parser.parse_args()

    metrics_log_dir = os.path.join(args.model_path, "eval_logs")
    os.makedirs(metrics_log_dir, exist_ok=True)
    rank_metrics_file = os.path.join(metrics_log_dir, f"run_sample_rank{rank}.jsonl")
    rank_samples_file = os.path.join(
        metrics_log_dir, f"run_sample_samples_rank{rank}.jsonl"
    )

    device = torch.device("cuda", local_rank)
    model, graph, noise = load_model(args.model_path, device)
    cfg = None
    hydra_cfg = os.path.join(args.model_path, ".hydra", "config.yaml")
    if os.path.exists(hydra_cfg):
        cfg = utils.load_hydra_config_from_run(args.model_path)

    model_cfg = getattr(model, "config", None)
    cfg = cfg or model_cfg
    worker_cfg = getattr(cfg, "worker", cfg) if cfg is not None else None

    length = args.length
    if length is None and cfg is not None:
        length = getattr(getattr(cfg, "model", None), "length", None)
    length = length or 1024

    steps = args.steps
    if steps is None and cfg is not None:
        steps = getattr(getattr(cfg, "sampling", None), "steps", None)
    steps = steps or 1024

    predictor = "analytic"
    denoise = True
    if cfg is not None:
        predictor = getattr(getattr(cfg, "sampling", None), "predictor", predictor)
        denoise = getattr(getattr(cfg, "sampling", None), "noise_removal", denoise)

    text_tokenizer_name = "gpt2"
    caption_tokenizer_name = "bert-base-uncased"
    if cfg is not None:
        text_tokenizer_name = getattr(
            getattr(cfg, "tokenizer", None), "text", text_tokenizer_name
        )
        caption_tokenizer_name = getattr(
            getattr(cfg, "tokenizer", None), "caption", caption_tokenizer_name
        )

    tokenizer_text = get_text_tokenizer(text_tokenizer_name)
    tokenizer_caption = get_caption_tokenizer(caption_tokenizer_name)

    if cfg is None or getattr(getattr(cfg, "data", None), "format", None) != "entry":
        raise ValueError("run_sample.py currently supports only entry-format datasets")

    valid_set = data_process.get_entry_dataset(
        cfg.data.validset.name,
        "validation",
        cache_dir=cfg.data.validset.cache_dir,
        text_max_length=cfg.data.max_length,
        num_proc=cfg.data.num_proc,
        text_tokenizer_name=text_tokenizer_name,
        caption_tokenizer_name=caption_tokenizer_name,
    )

    if distributed:
        shard_indices = list(range(rank, len(valid_set), world_size))
        valid_set = Subset(valid_set, shard_indices)

    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    def truncate_at_eos(batch_ids, eos_id):
        output = []
        for row in batch_ids.tolist():
            if eos_id in row:
                k = row.index(eos_id)
                output.append(row[:k])
            else:
                output.append(row)
        return output

    metric = get_alignment_metric(
        model_name="intfloat/e5-base-v2",
        use_sentence_transformers=True,
        device=str(device),
    )

    eval_cfg = getattr(cfg, "eval", None) if cfg is not None else None
    enable_mauve = bool(getattr(eval_cfg, "mauve", True))
    mauve_max_text_length = int(getattr(eval_cfg, "mauve_max_text_length", 256))

    def extract_body(sentences):
        result = []
        for s in sentences:
            if "Body:" in s:
                result.append(s.split("Body:", 1)[1].lstrip())
            else:
                result.append(s)
        return result

    with (
        open(rank_metrics_file, "w", encoding="utf-8", buffering=1) as rank_log_fp,
        open(rank_samples_file, "w", encoding="utf-8", buffering=1) as sample_log_fp,
        torch.inference_mode(),
    ):
        eval_model = get_eval_lm("gpt2-large", device)

        batch_size = int(
            getattr(getattr(worker_cfg, "eval", None), "perplexity_batch_size", 32)
        )

        similarity_sum = 0.0
        similarity_count = 0
        all_generated_texts = []
        all_reference_texts = []

        total_loss = 0.0
        total_tokens = 0.0

        eos_token_id = tokenizer_text.eos_token_id

        eval_iterator = valid_loader
        if rank == 0:
            eval_iterator = tqdm(valid_loader, desc="Evaluating", unit="batch")

        for batch_idx, eval_batch_data in enumerate(eval_iterator):
            eval_text_mask = eval_batch_data["text_attention_mask"].to(device)
            eval_text = eval_batch_data.get("text_input_ids")
            eval_style_caption = eval_batch_data["style_caption_input_ids"].to(device)
            eval_style_caption_mask = eval_batch_data[
                "style_caption_attention_mask"
            ].to(device)

            local_length = min(length, eval_text_mask.shape[1])

            local_batch_size = eval_text_mask.size(0)
            if local_batch_size == 0:
                continue

            sampling_fn = sampling.PCSampler(
                graph,
                noise,
                (local_batch_size, local_length),
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
                args.cfg_scale,
            )

            sample_trunc = truncate_at_eos(samples, tokenizer_text.eos_token_id)
            sentences = tokenizer_text.batch_decode(sample_trunc)
            generated_bodies = extract_body(sentences)
            captions = tokenizer_caption.batch_decode(
                eval_style_caption, skip_special_tokens=True
            )
            decoded_references = None
            if eval_text is not None:
                decoded_references = tokenizer_text.batch_decode(
                    eval_text, skip_special_tokens=True
                )

            for local_idx, (caption, generated_text) in enumerate(
                zip(captions, generated_bodies)
            ):
                sample_record = {
                    "batch_idx": int(batch_idx),
                    "rank": int(rank),
                    "sample_idx_in_batch": int(local_idx),
                    "caption": caption,
                    "generated_text": generated_text,
                }
                if decoded_references is not None and local_idx < len(
                    decoded_references
                ):
                    sample_record["reference_text"] = decoded_references[local_idx]
                sample_log_fp.write(
                    json.dumps(sample_record, ensure_ascii=False) + "\n"
                )
            sample_log_fp.flush()

            if len(sentences) > 0:
                similarity_scores = metric.score_batch(captions, generated_bodies)
                similarity_sum += float(sum(similarity_scores))
                similarity_count += len(similarity_scores)
                batch_avg_similarity = sum(similarity_scores) / len(similarity_scores)
            else:
                batch_avg_similarity = float("nan")

            if enable_mauve and eval_text is not None and len(sentences) > 0:
                all_generated_texts.extend(generated_bodies)
                all_reference_texts.extend(decoded_references)

            num_samples = samples.size(0)
            batch_loss_total = torch.zeros(1, device=device)
            batch_tokens_total = torch.zeros(1, device=device)
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                s = samples[start:end]  # (b, T)

                if s.size(0) == 0:
                    continue

                outputs = eval_model(s)
                logits = outputs.logits  # (b, T, V)

                logits = logits[:, :-1, :].contiguous()  # (b, T-1, V)
                targets = s[:, 1:]  # (b, T-1)

                vocab_size = logits.size(-1)
                logits_flat = logits.view(-1, vocab_size)
                targets_flat = targets.reshape(-1)

                token_losses = F.cross_entropy(
                    logits_flat,
                    targets_flat,
                    reduction="none",
                ).view_as(
                    targets
                )  # (b, T-1)

                eos_mask = (targets == eos_token_id).int()  # (b, T-1)
                cumsum = eos_mask.cumsum(dim=-1)
                valid_mask = (cumsum == 0).to(token_losses.dtype)

                masked_losses = token_losses * valid_mask

                batch_loss_sum = masked_losses.sum()
                batch_token_sum = valid_mask.sum()

                batch_loss_total += batch_loss_sum
                batch_tokens_total += batch_token_sum

            total_loss += float(batch_loss_total.item())
            total_tokens += float(batch_tokens_total.item())

            batch_tokens_for_ppl = batch_tokens_total.clamp_min(1)
            batch_ppl = torch.exp(batch_loss_total / batch_tokens_for_ppl).item()

            rank_log_fp.write(
                json.dumps(
                    {
                        "batch_idx": int(batch_idx),
                        "rank": int(rank),
                        "local_batch_size": int(local_batch_size),
                        "avg_similarity": float(batch_avg_similarity),
                        "mauve": None,
                        "perplexity": float(batch_ppl),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            rank_log_fp.flush()

            logger.info(
                "[rank %d] Batch %d/%d | avg_similarity=%.4f | perplexity=%.4f | mauve=deferred",
                rank,
                batch_idx + 1,
                len(valid_loader),
                float(batch_avg_similarity),
                float(batch_ppl),
            )

        if distributed:
            local_stats = {
                "similarity_sum": similarity_sum,
                "similarity_count": similarity_count,
                "total_loss": total_loss,
                "total_tokens": total_tokens,
            }
            gathered_stats = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_stats, local_stats)

            similarity_sum = sum(s["similarity_sum"] for s in gathered_stats)
            similarity_count = sum(s["similarity_count"] for s in gathered_stats)
            total_loss = sum(s["total_loss"] for s in gathered_stats)
            total_tokens = sum(s["total_tokens"] for s in gathered_stats)

        avg_similarity = similarity_sum / max(similarity_count, 1)
        avg_mauve = float("nan")

        if enable_mauve:
            all_refs = all_reference_texts
            all_gens = all_generated_texts
            if distributed:
                gathered_refs = [None for _ in range(world_size)]
                gathered_gens = [None for _ in range(world_size)]
                dist.all_gather_object(gathered_refs, all_reference_texts)
                dist.all_gather_object(gathered_gens, all_generated_texts)

                all_refs = []
                all_gens = []
                for refs in gathered_refs:
                    all_refs.extend(refs)
                for gens in gathered_gens:
                    all_gens.extend(gens)

            if rank == 0 and len(all_refs) > 0 and len(all_gens) > 0:
                avg_mauve = get_mauve_score(
                    p_texts=all_refs,
                    q_texts=all_gens,
                    device_id=local_rank if device.type == "cuda" else -1,
                    max_text_length=mauve_max_text_length,
                    verbose=False,
                )

        total_tokens = max(total_tokens, 1.0)

        mean_loss = total_loss / total_tokens
        perplexity = math.exp(mean_loss)

    if rank == 0:
        print(
            "Average similarity score between captions and generated text: "
            f"{avg_similarity:.4f}"
        )
        if enable_mauve and avg_mauve == avg_mauve:
            print(f"Average MAUVE score: {avg_mauve:.4f}")
        else:
            print("Average MAUVE score: nan (mauve disabled or no valid batches)")
        print(f"Perplexity of generated text: {perplexity:.4f}")

    with open(rank_metrics_file, "a", encoding="utf-8", buffering=1) as rank_log_fp:
        rank_log_fp.write(
            json.dumps(
                {
                    "record_type": "summary",
                    "rank": int(rank),
                    "avg_similarity": float(avg_similarity),
                    "avg_mauve": (
                        float(avg_mauve)
                        if (enable_mauve and avg_mauve == avg_mauve)
                        else None
                    ),
                    "perplexity": float(perplexity),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        rank_log_fp.flush()

    if distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
