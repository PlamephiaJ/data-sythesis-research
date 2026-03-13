from functools import lru_cache

from transformers import BertTokenizerFast, GPT2TokenizerFast

from utils import hf_local


@lru_cache(maxsize=None)
def get_text_tokenizer(name: str = "gpt2") -> GPT2TokenizerFast:
    tokenizer_path = hf_local.resolve_pretrained_path(name)
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@lru_cache(maxsize=None)
def get_caption_tokenizer(name: str = "bert-base-uncased") -> BertTokenizerFast:
    tokenizer_path = hf_local.resolve_pretrained_path(name)
    return BertTokenizerFast.from_pretrained(tokenizer_path, local_files_only=True)
