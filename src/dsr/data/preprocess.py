"""Dataset preprocessing utilities and tokenizer helpers."""

from __future__ import annotations

import json
import re
from itertools import chain
from typing import Callable, Dict, Iterable

import requests
from datasets import Dataset, load_dataset
from transformers import GPT2TokenizerFast


def wt_detokenizer(string: str) -> str:
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string


def ptb_detokenizer(text: str) -> str:
    text = text.replace(" 's", "'s")
    text = text.replace("s ' ", "s' ")
    text = text.replace(" n't", "n't")
    text = text.replace(" \n ", "\n")
    text = text.replace("\\/", "/")
    for _ in range(10):
        text = text.replace(" N ", " 1 ")
    text = text.replace("$ 1", "$1")
    text = text.replace("# 1", "#1")
    text = text.replace("<unk>", "?")
    return text


def lm1b_detokenizer(text: str) -> str:
    text = text.replace("http : / / ", "http://")
    text = text.replace("https : / / ", "https://")
    text = re.sub(r" \'(\w+)", r"'\1", text)
    text = re.sub(r" (\w+) \. ", r" \1. ", text)
    text = re.sub(r" (\w+) \.$", r" \1.", text)
    text = text.replace(" ? ", "? ")
    text = re.sub(r" \?$", "?", text)
    text = text.replace(" ! ", "! ")
    text = re.sub(r" \!$", "!", text)
    text = text.replace(" , ", ", ")
    text = text.replace(" : ", ": ")
    text = text.replace(" ; ", "; ")
    text = text.replace(" / ", "/")
    text = re.sub(r"\" ([^\"]+) \"", r'"\1"', text)
    text = re.sub(r"\' ([^\']+) \'", r"'\1'", text)
    text = re.sub(r"\( ([^\(\)]+) \)", r"(\1)", text)
    text = re.sub(r"\[ ([^\[\]]+) \]", r"[\1]", text)
    text = text.replace("$ ", "$")
    text = text.replace("£ ", "£")
    return text


def lambada_detokenizer(text: str) -> str:
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    return "\n" + text.strip()


def get_lambada_test_dataset() -> Dataset:
    url = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"

    def read_jsonl_to_list(stream_url: str) -> Iterable[Dict]:
        response = requests.get(stream_url, stream=True, timeout=30)
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if line:
                yield json.loads(line)

    lambada_data = list(read_jsonl_to_list(url))
    dataset = Dataset.from_list(lambada_data)
    return dataset


def _apply_detokenizer(detokenizer: Callable[[str], str]):
    def detok(text):
        for i, value in enumerate(text, 0):
            text[i] = detokenizer(value)
        return text

    return detok


def _detokenizer_for(name: str) -> Callable[[str], str] | None:
    if name.startswith("wikitext"):
        return wt_detokenizer
    if name == "ptb":
        return ptb_detokenizer
    if name == "lm1b":
        return lm1b_detokenizer
    if name == "lambada":
        return lambada_detokenizer
    return None


def get_dataset(name: str, mode: str, cache_dir: str | None = None, block_size: int = 1024):
    if name == "wikitext103":
        dataset = load_dataset("wikitext", name="wikitext-103-raw-v1", cache_dir=cache_dir)
    elif name == "wikitext2":
        dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", cache_dir=cache_dir)
    elif name == "ptb":
        dataset = load_dataset("ptb_text_only", cache_dir=cache_dir)
    elif name == "lambada":
        dataset = get_lambada_test_dataset()
    else:
        dataset = load_dataset(name, cache_dir=cache_dir)

    data = dataset if name == "lambada" else dataset[mode]
    detokenizer = _detokenizer_for(name)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    eos_token = tokenizer.encode(tokenizer.eos_token)[0]

    def preprocess_and_tokenize(example):
        text = example["sentence"] if name == "ptb" else example["text"]
        if detokenizer is not None:
            text = _apply_detokenizer(detokenizer)(text)

        tokens = tokenizer(text, return_attention_mask=False)
        for token in tokens["input_ids"]:
            token.append(eos_token)
        return tokens

    tokenized_dataset = data.map(preprocess_and_tokenize, batched=True, num_proc=120, load_from_cache_file=True)
    remove_key = "sentence" if name == "ptb" else "text"
    tokenized_dataset = tokenized_dataset.remove_columns(remove_key)

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    chunked_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=120, load_from_cache_file=True)
    return chunked_dataset.with_format("torch")
