from functools import lru_cache

from transformers import BertTokenizerFast, GPT2TokenizerFast


@lru_cache(maxsize=None)
def get_text_tokenizer(name: str = "gpt2") -> GPT2TokenizerFast:
    tokenizer = GPT2TokenizerFast.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@lru_cache(maxsize=None)
def get_caption_tokenizer(name: str = "bert-base-uncased") -> BertTokenizerFast:
    return BertTokenizerFast.from_pretrained(name)
