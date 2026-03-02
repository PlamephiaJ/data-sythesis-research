import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_process.data import get_entry_dataset
from utils.eval_factory import get_eval_lm
from utils.tokenizer_factory import get_text_tokenizer


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"


if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@torch.inference_mode()
def compute_ppl(
    dataset, eval_model, eos_token_id: int, device: torch.device, batch_size: int
):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_loss = torch.zeros(1, device=device)
    total_tokens = torch.zeros(1, device=device)

    for batch in loader:
        s = batch["text_input_ids"].to(device)

        outputs = eval_model(s)
        logits = outputs.logits

        logits = logits[:, :-1, :].contiguous()
        targets = s[:, 1:]

        vocab_size = logits.size(-1)
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.reshape(-1)

        token_losses = F.cross_entropy(
            logits_flat,
            targets_flat,
            reduction="none",
        ).view_as(targets)

        eos_mask = (targets == eos_token_id).int()
        cumsum = eos_mask.cumsum(dim=-1)
        valid_mask = (cumsum == 0).to(token_losses.dtype)

        masked_losses = token_losses * valid_mask

        total_loss += masked_losses.sum()
        total_tokens += valid_mask.sum()

    total_tokens = total_tokens.clamp_min(1)
    mean_loss = total_loss / total_tokens
    ppl = mean_loss.exp()
    return {
        "total_tokens": int(total_tokens.item()),
        "mean_loss": float(mean_loss.item()),
        "ppl": float(ppl.item()),
    }


def build_args():
    parser = argparse.ArgumentParser(
        description="Compute entry dataset (phish-email) PPL using run_train.py style-control method."
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "both"],
        default="both",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data_phish/json",
        help="Path to phish-email JSON dataset directory.",
    )
    parser.add_argument(
        "--text-max-length",
        type=int,
        default=1024,
        help="Max text length for entry dataset tokenization.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=120,
        help="Num processes used in HF dataset map/filter.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="PPL eval batch size (same role as worker.eval.perplexity_batch_size).",
    )
    parser.add_argument(
        "--eval-model",
        type=str,
        default="gpt2-large",
        help="Evaluation language model name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device, e.g. cuda, cuda:0, cpu.",
    )
    return parser.parse_args()


def main():
    args = build_args()

    device = torch.device(args.device)
    tokenizer_text = get_text_tokenizer("gpt2")
    eos_token_id = tokenizer_text.eos_token_id

    eval_model = get_eval_lm(args.eval_model, device)

    splits = ["train", "validation"] if args.split == "both" else [args.split]

    print(f"device={device}")
    print(f"eval_model={args.eval_model}")
    print(f"cache_dir={args.cache_dir}")
    print(f"text_max_length={args.text_max_length}")
    print(f"batch_size={args.batch_size}")

    for split in splits:
        dataset = get_entry_dataset(
            name="phish-email",
            mode=split,
            cache_dir=args.cache_dir,
            text_max_length=args.text_max_length,
            num_proc=args.num_proc,
            text_tokenizer_name="gpt2",
            caption_tokenizer_name="bert-base-uncased",
        )

        metrics = compute_ppl(
            dataset=dataset,
            eval_model=eval_model,
            eos_token_id=eos_token_id,
            device=device,
            batch_size=args.batch_size,
        )

        print(
            f"[{split}] samples={len(dataset)}, total_tokens={metrics['total_tokens']}, "
            f"mean_loss={metrics['mean_loss']:.6f}, ppl={metrics['ppl']:.6f}"
        )


if __name__ == "__main__":
    main()
