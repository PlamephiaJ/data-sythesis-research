from itertools import chain

import numpy as np
from torch.utils.data import DataLoader, DistributedSampler

import data_process.dataset_factory as dataset_factory
from data_process.entry_processor_factory import get_entry_pipeline
from utils.tokenizer_factory import get_text_tokenizer

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

    tokenizer = get_text_tokenizer("gpt2")
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


def get_entry_dataset(
    name,
    mode,
    cache_dir=None,
    text_max_length=1024,
    caption_max_length=256,
    num_proc=120,
    text_tokenizer_name="gpt2",
    caption_tokenizer_name="bert-base-uncased",
):
    data = dataset_factory.get_dataset(name, mode=mode, cache_dir=cache_dir)
    try:
        pipeline = get_entry_pipeline(
            name,
            text_max_length=text_max_length,
            caption_max_length=caption_max_length,
            text_tokenizer_name=text_tokenizer_name,
            caption_tokenizer_name=caption_tokenizer_name,
        )
    except KeyError as exc:
        raise NotImplementedError(f"Entry dataset {name} not implemented yet.") from exc

    data = data.filter(pipeline.validate_example, num_proc=num_proc)

    tokenized_dataset = data.map(
        pipeline.preprocess_batch,
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=True,
    )

    removable_columns = [
        col
        for col in pipeline.columns_to_remove
        if col in tokenized_dataset.column_names
    ]
    if removable_columns:
        tokenized_dataset = tokenized_dataset.remove_columns(removable_columns)

    tokenized_dataset = tokenized_dataset.with_format("torch")
    return tokenized_dataset


def get_dataloaders(config, distributed=True):
    if "worker" not in config:
        raise ValueError("Missing required config key: worker")
    worker_cfg = config.worker

    if (
        worker_cfg.training.batch_size % (worker_cfg.ngpus * worker_cfg.training.accum)
        != 0
    ):
        raise ValueError(
            f"Train Batch Size {worker_cfg.training.batch_size} is not divisible by {worker_cfg.ngpus} gpus with accumulation {worker_cfg.training.accum}."
        )
    if worker_cfg.eval.batch_size % (worker_cfg.ngpus * worker_cfg.training.accum) != 0:
        raise ValueError(
            f"Eval Batch Size for {worker_cfg.eval.batch_size} is not divisible by {worker_cfg.ngpus} gpus with accumulation {worker_cfg.training.accum}."
        )

    if "format" not in config.data.trainset:
        raise ValueError(
            "Missing required config key: data.trainset.format. Please set it explicitly to 'chunk' or 'entry'."
        )
    if "format" not in config.data.validset:
        raise ValueError(
            "Missing required config key: data.validset.format. Please set it explicitly to 'chunk' or 'entry'."
        )

    train_format = config.data.trainset.format
    valid_format = config.data.validset.format
    if train_format != valid_format:
        raise ValueError(
            f"Mismatched dataset formats: trainset={train_format}, validset={valid_format}. "
            "Both must be the same for current dataloader pipeline."
        )

    data_format = train_format

    if data_format == "chunk":
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
    elif data_format == "entry":
        train_set = get_entry_dataset(
            config.data.trainset.name,
            "train",
            cache_dir=config.data.trainset.cache_dir,
            text_max_length=config.data.max_length,
            num_proc=config.data.num_proc,
            text_tokenizer_name=config.tokenizer.text,
            caption_tokenizer_name=config.tokenizer.caption,
        )
        valid_set = get_entry_dataset(
            config.data.validset.name,
            "validation",
            cache_dir=config.data.validset.cache_dir,
            text_max_length=config.data.max_length,
            num_proc=config.data.num_proc,
            text_tokenizer_name=config.tokenizer.text,
            caption_tokenizer_name=config.tokenizer.caption,
        )
    else:
        raise ValueError(
            f"Unknown data format {data_format}, must be 'chunk' for language modeling or 'entry' for conditioned generation."
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
            batch_size=worker_cfg.training.batch_size
            // (worker_cfg.ngpus * worker_cfg.training.accum),
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
            batch_size=worker_cfg.eval.batch_size
            // (worker_cfg.ngpus * worker_cfg.training.accum),
            sampler=test_sampler,
            num_workers=4,
            pin_memory=True,
            shuffle=(test_sampler is None),
        )
    )
    return train_loader, valid_loader
