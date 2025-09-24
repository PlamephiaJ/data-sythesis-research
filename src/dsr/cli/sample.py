"""CLI for sampling text from trained models."""

import argparse
from typing import Callable, Optional

import torch
from transformers import GPT2TokenizerFast

from ..train import sampler
from .load_model import load_model


def _build_projection_fn(
    tokenizer: GPT2TokenizerFast,
    prefix: Optional[str],
    suffix: Optional[str],
    sequence_length: int,
    batch_size: int,
    device: torch.device,
) -> Callable[[torch.Tensor], torch.Tensor]:
    if not prefix and not suffix:
        return lambda x: x

    token_ids = []
    positions = []

    if prefix:
        prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
        if len(prefix_ids) > sequence_length:
            raise ValueError("Prefix is longer than the configured sequence length.")
        token_ids.extend(prefix_ids)
        positions.extend(range(len(prefix_ids)))

    if suffix:
        suffix_ids = tokenizer(suffix, add_special_tokens=False).input_ids
        if len(suffix_ids) > sequence_length:
            raise ValueError("Suffix is longer than the configured sequence length.")
        start = sequence_length - len(suffix_ids)
        positions.extend(range(start, sequence_length))
        token_ids.extend(suffix_ids)

    if len(set(positions)) != len(positions):
        raise ValueError("Prefix and suffix assignments overlap in the target sequence.")

    index_tensor = torch.tensor(positions, device=device, dtype=torch.long)
    value_tensor = torch.tensor(token_ids, device=device, dtype=torch.long)
    value_tensor = value_tensor.unsqueeze(0).repeat(batch_size, 1)

    def proj(x: torch.Tensor) -> torch.Tensor:
        x[:, index_tensor] = value_tensor
        return x

    return proj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate samples using a trained model.")
    parser.add_argument(
        "--model_path", default="louaaron/sedd-medium", type=str, help="Model identifier or local path"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Number of samples to draw")
    parser.add_argument("--steps", type=int, default=1024, help="Number of predictor-corrector steps")
    parser.add_argument("--sequence_length", type=int, default=1024, help="Sequence length expected by the model")
    parser.add_argument("--predictor", type=str, default="analytic", help="Registered predictor to use")
    parser.add_argument("--prefix", type=str, default=None, help="Optional prefix constraint applied during sampling")
    parser.add_argument("--suffix", type=str, default=None, help="Optional suffix constraint applied during sampling")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, graph, noise = load_model(args.model_path, device)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    projection_fn = _build_projection_fn(
        tokenizer,
        prefix=args.prefix,
        suffix=args.suffix,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        device=device,
    )

    sampling_fn = sampler.get_pc_sampler(
        graph=graph,
        noise=noise,
        batch_dims=(args.batch_size, args.sequence_length),
        predictor=args.predictor,
        steps=args.steps,
        device=device,
        proj_fun=projection_fn,
    )

    samples = projection_fn(sampling_fn(model))
    text_samples = tokenizer.batch_decode(samples)

    separator = "=" * 49
    for text in text_samples:
        print(text)
        print(separator)


if __name__ == "__main__":
    main()
