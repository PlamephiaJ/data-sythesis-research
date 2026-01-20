from functools import lru_cache
from typing import Optional

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
