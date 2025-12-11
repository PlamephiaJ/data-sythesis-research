from itertools import chain

import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
from transformers import GPT2TokenizerFast

import data_process.dataset_factory as dataset_factory

from .detokenizer_factory import get_detokenizer


def cycle_loader(dataloader, sampler=None):
    while 1:
        if sampler is not None:
            sampler.set_epoch(np.random.randint(0, 100000))
        for data in dataloader:
            yield data


def get_chunk_dataset(name, mode, cache_dir=None, block_size=1024, num_proc=120):

    data = dataset_factory.get_dataset(name, mode=mode, cache_dir=cache_dir)

    try:
        detokenizer = get_detokenizer(name)
    except KeyError:
        detokenizer = None

    def _apply_detokenizer(detokenizer):
        def detok(text):
            for i, t in enumerate(text, 0):
                text[i] = detokenizer(t)
            return text

        return detok

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    EOS = tokenizer.encode(tokenizer.eos_token)[0]

    def preprocess_and_tokenize(example):
        if name == "ptb":
            text = example["sentence"]
        else:
            text = example["text"]
        # print(list(example.keys()))
        # exit()

        if detokenizer is not None:
            text = _apply_detokenizer(detokenizer)(text)

        tokens = tokenizer(text, return_attention_mask=False)
        # add in EOS token following
        # https://github.com/jcpeterson/openwebtext/blob/master/tokenize_text.py#L67
        for token in tokens["input_ids"]:
            token.append(EOS)
        return tokens

    tokenized_dataset = data.map(
        preprocess_and_tokenize,
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=True,
    )
    if name == "ptb":
        tokenized_dataset = tokenized_dataset.remove_columns("sentence")
    else:
        tokenized_dataset = tokenized_dataset.remove_columns("text")

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    chunked_dataset = tokenized_dataset.map(
        group_texts, batched=True, num_proc=num_proc, load_from_cache_file=True
    )
    chunked_dataset = chunked_dataset.with_format("torch")

    return chunked_dataset


def get_entry_dataset(name, mode, cache_dir=None, max_length=1024, num_proc=120):
    data = dataset_factory.get_dataset(name, mode=mode, cache_dir=cache_dir)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    EOS = tokenizer.encode(tokenizer.eos_token)[0]

    def preprocess_and_tokenize(example):
        if name == "ptb":
            text = example["sentence"]
        else:
            text = example["text"]

        tokens = tokenizer(text, return_attention_mask=False)
        # add in EOS token following
        for token in tokens["input_ids"]:
            token.append(EOS)
        return tokens

    tokenized_dataset = data.map(
        preprocess_and_tokenize,
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=True,
    )
    if name == "ptb":
        tokenized_dataset = tokenized_dataset.remove_columns("sentence")
    else:
        tokenized_dataset = tokenized_dataset.remove_columns("text")

    tokenized_dataset = tokenized_dataset.filter(
        lambda example: all(len(ids) <= max_length for ids in example["input_ids"]),
        num_proc=num_proc,
    )

    tokenized_dataset = tokenized_dataset.with_format("torch")

    return tokenized_dataset


def get_dataloaders(config, distributed=True):
    if config.training.batch_size % (config.ngpus * config.training.accum) != 0:
        raise ValueError(
            f"Train Batch Size {config.training.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}."
        )
    if config.eval.batch_size % (config.ngpus * config.training.accum) != 0:
        raise ValueError(
            f"Eval Batch Size for {config.eval.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}."
        )

    if config.data.format == "chunk":
        train_set = get_chunk_dataset(
            config.data.trainset.name,
            "train",
            cache_dir=config.data.trainset.cache_dir,
            block_size=config.model.length,
            num_proc=config.data.num_proc,
        )
        valid_set = get_chunk_dataset(
            config.data.validset.name,
            "validation" if config.data.validset.name != "text8" else "test",
            cache_dir=config.data.validset.cache_dir,
            block_size=config.model.length,
            num_proc=config.data.num_proc,
        )
    elif config.data.format == "entry":
        train_set = get_entry_dataset(
            config.data.train,
            "train",
            cache_dir=config.data.cache_dir,
            max_length=config.data.max_length,
            num_proc=config.data.num_proc,
        )
        valid_set = get_entry_dataset(
            config.data.valid,
            "validation" if config.data.valid != "text8" else "test",
            cache_dir=config.data.cache_dir,
            max_length=config.data.max_length,
            num_proc=config.data.num_proc,
        )
    else:
        raise ValueError(
            f"Unknown data format {config.data.format}, must be 'chunk' for language modeling or 'entry' for conditioned generation."
        )

    if distributed:
        train_sampler = DistributedSampler(train_set)
        test_sampler = DistributedSampler(valid_set)
    else:
        train_sampler = None
        test_sampler = None

    train_loader = cycle_loader(
        DataLoader(
            train_set,
            batch_size=config.training.batch_size
            // (config.ngpus * config.training.accum),
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
            shuffle=(train_sampler is None),
            persistent_workers=True,
        )
    )
    valid_loader = cycle_loader(
        DataLoader(
            valid_set,
            batch_size=config.eval.batch_size // (config.ngpus * config.training.accum),
            sampler=test_sampler,
            num_workers=4,
            pin_memory=True,
            shuffle=(test_sampler is None),
        )
    )
    return train_loader, valid_loader
