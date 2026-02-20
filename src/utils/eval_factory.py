from functools import lru_cache
from typing import Optional

import mauve
import torch
from transformers import GPT2LMHeadModel

from metric import aliment


@lru_cache(maxsize=None)
def get_eval_lm(model_name: str = "gpt2-large", device: str = "cuda"):
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model = model.to(torch.device(device)).eval()
    return model


@lru_cache(maxsize=None)
def get_alignment_metric(
    model_name: str = "intfloat/e5-base-v2",
    use_sentence_transformers: bool = True,
    device: Optional[str] = None,
):
    return aliment.make_default_alignment_metric(
        model_name=model_name,
        use_sentence_transformers=use_sentence_transformers,
        device=device,
        policy=aliment.MaxSimPolicy(),
    )


def get_mauve_score(
    p_texts,
    q_texts,
    device_id: int,
    max_text_length: int,
    verbose: bool,
) -> float:
    if not p_texts or not q_texts:
        return float("nan")

    out = mauve.compute_mauve(
        p_text=list(p_texts),
        q_text=list(q_texts),
        device_id=device_id,
        max_text_length=max_text_length,
        verbose=verbose,
    )
    return float(out.mauve)
