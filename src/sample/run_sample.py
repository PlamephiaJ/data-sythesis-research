import argparse
import logging

import torch

import sample.sampling as sampling
from sample.load_model import load_model
from utils.tokenizer_factory import get_text_tokenizer


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--model_path", default="louaaron/sedd-medium", type=str)
    parser.add_argument("--dataset", default="wikitext103", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1024)
    args = parser.parse_args()

    device = torch.device("cuda")
    model, graph, noise = load_model(args.model_path, device)
    tokenizer = get_text_tokenizer("gpt2")

    sampling_fn = sampling.get_pc_sampler(
        graph, noise, (args.batch_size, 1024), "analytic", args.steps, device=device
    )

    samples = sampling_fn(model)

    text_samples = tokenizer.batch_decode(samples)
    for i in text_samples:
        logger.info(i)
        logger.info("=================================================")


if __name__ == "__main__":
    main()
