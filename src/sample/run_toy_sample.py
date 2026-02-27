import argparse
import json
import logging
import os

import torch

import sample.sampling as sampling
import utils.utils as utils
from sample.load_model import load_model
from utils.tokenizer_factory import get_caption_tokenizer, get_text_tokenizer


def truncate_at_eos(batch_ids, eos_id):
    output = []
    for row in batch_ids.tolist():
        if eos_id in row:
            k = row.index(eos_id)
            output.append(row[:k])
        else:
            output.append(row)
    return output


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Toy sampler: generate emails from a description"
    )
    parser.add_argument(
        "--model_path", default="exp_local/phish-email/2026.02.02/105439/", type=str
    )
    parser.add_argument(
        "--description",
        "-d",
        required=True,
        help="Description / caption to condition on",
    )
    parser.add_argument(
        "--num", "-n", type=int, default=4, help="Number of samples to generate"
    )
    parser.add_argument(
        "--length", type=int, default=1024, help="Generation length (tokens)"
    )
    parser.add_argument(
        "--steps", type=int, default=128, help="Sampling steps override"
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=0.5,
        help="Classifier-free guidance scale, higher values increase faithfulness to the description but may reduce diversity",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use, e.g. cuda:0 or cpu"
    )
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model, graph, noise = load_model(args.model_path, device)

    cfg = None
    hydra_cfg = os.path.join(args.model_path, ".hydra", "config.yaml")
    if os.path.exists(hydra_cfg):
        cfg = utils.load_hydra_config_from_run(args.model_path)

    # tokenizer names from cfg or defaults
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

    length = args.length
    if length is None and cfg is not None:
        length = getattr(getattr(cfg, "model", None), "length", None)
    length = int(length or 1024)

    steps = args.steps
    if steps is None and cfg is not None:
        steps = getattr(getattr(cfg, "sampling", None), "steps", None)
    steps = int(steps or 128)

    predictor = "analytic"
    denoise = True
    if cfg is not None:
        predictor = getattr(getattr(cfg, "sampling", None), "predictor", predictor)
        denoise = getattr(getattr(cfg, "sampling", None), "noise_removal", denoise)

    n = int(args.num)

    # tokenize caption / description
    cap_max_len = min(128, tokenizer_caption.model_max_length or 128)
    cap_enc = tokenizer_caption(
        args.description,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=cap_max_len,
    )
    style_ids = cap_enc["input_ids"]  # (1, Lc)
    style_mask = cap_enc["attention_mask"]

    # repeat to batch size n
    style_ids = style_ids.repeat(n, 1).to(device)
    style_mask = style_mask.repeat(n, 1).to(device)

    # context mask: mark all positions as valid (no padding), shape (n, length)
    # Use ones so FlashAttention varlen sees non-zero lengths.
    x_mask = torch.ones((n, length), dtype=torch.long, device=device)

    sampling_fn = sampling.PCSampler(
        graph, noise, (n, length), predictor, steps, denoise, 1e-5, device
    )

    with torch.inference_mode():
        samples = sampling_fn(
            model, x_mask, style_ids, style_mask, float(args.cfg_scale)
        )

    # truncate and decode
    sample_trunc = truncate_at_eos(samples, tokenizer_text.eos_token_id)
    sentences = tokenizer_text.batch_decode(sample_trunc, skip_special_tokens=True)

    for i, s in enumerate(sentences):
        out = {"index": i, "description": args.description, "generated_text": s}
        print("-" * 80)
        print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
