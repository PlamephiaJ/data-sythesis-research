import argparse
import logging
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sample.sampling as sampling
from data_process import data as data_process
from sample.load_model import load_model
from utils import utils
from utils.eval_factory import get_alignment_metric, get_eval_lm
from utils.tokenizer_factory import get_caption_tokenizer, get_text_tokenizer


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument(
        "--model_path", default="exp_local/phish-email/2026.02.02/105439/", type=str
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--length", type=int, default=None)
    parser.add_argument("--cfg_scale", type=float, default=0.0)
    args = parser.parse_args()

    device = torch.device("cuda")
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

        sampling_shape = (
            worker_cfg.training.batch_size
            // (worker_cfg.ngpus * worker_cfg.training.accum),
            cfg.model.length,
        )

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

    if (
        cfg is not None
        and getattr(getattr(cfg, "data", None), "format", None) == "entry"
    ):
        valid_set = data_process.get_entry_dataset(
            cfg.data.validset.name,
            "validation",
            cache_dir=cfg.data.validset.cache_dir,
            text_max_length=cfg.data.max_length,
            num_proc=cfg.data.num_proc,
            text_tokenizer_name=text_tokenizer_name,
            caption_tokenizer_name=caption_tokenizer_name,
        )
        valid_loader = DataLoader(
            valid_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        eval_iter = iter(valid_loader)
        eval_batch_data = next(eval_iter)
        # eval_text = eval_batch_data["text_input_ids"][: sampling_shape[0]].to(device)
        eval_text_mask = eval_batch_data["text_attention_mask"][: sampling_shape[0]].to(
            device
        )
        eval_style_caption = eval_batch_data["style_caption_input_ids"][
            : sampling_shape[0]
        ].to(device)
        eval_style_caption_mask = eval_batch_data["style_caption_attention_mask"][
            : sampling_shape[0]
        ].to(device)
        length = min(length, eval_text_mask.shape[1])

    sampling_fn = sampling.PCSampler(
        graph,
        noise,
        (args.batch_size, length),
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

    def truncate_at_eos(batch_ids, eos_id):
        output = []
        for row in batch_ids.tolist():
            if eos_id in row:
                k = row.index(eos_id)
                output.append(row[:k])
            else:
                output.append(row)
        return output

    sample_trunc = truncate_at_eos(samples, tokenizer_text.eos_token_id)
    sentences = tokenizer_text.batch_decode(sample_trunc)
    captions = tokenizer_caption.batch_decode(
        eval_style_caption, skip_special_tokens=True
    )

    for idx, sample in enumerate(sentences):
        if captions is not None:
            logger.info("Caption: %s", captions[idx])
        logger.info(sample)
        logger.info("=================================================")

    metric = get_alignment_metric(
        model_name="intfloat/e5-base-v2",
        use_sentence_transformers=True,
        device=str(device),
    )

    def extract_body(sentences):
        result = []
        for s in sentences:
            if "Body:" in s:
                result.append(s.split("Body:", 1)[1].lstrip())
            else:
                result.append(s)
        return result

    similarity_scores = metric.score_batch(captions, extract_body(sentences))
    avg_similarity = sum(similarity_scores) / len(similarity_scores)

    print(
        f"Average similarity score between captions and generated text: {avg_similarity:.4f}"
    )

    with torch.inference_mode():
        eval_model = get_eval_lm("gpt2-large", device)

        batch_size = int(
            getattr(getattr(worker_cfg, "eval", None), "perplexity_batch_size", 32)
        )
        num_samples = sample.size(0)

        total_loss = torch.zeros(1, device=device)
        total_tokens = torch.zeros(1, device=device)

        eos_token_id = tokenizer_text.eos_token_id

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            s = sample[start:end]  # (b, T)

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

            total_loss += batch_loss_sum
            total_tokens += batch_token_sum

        total_tokens = total_tokens.clamp_min(1)

        mean_loss = total_loss / total_tokens
        perplexity = torch.exp(mean_loss)

    print(f"Perplexity of generated text: {perplexity.item():.4f}")


if __name__ == "__main__":
    main()
