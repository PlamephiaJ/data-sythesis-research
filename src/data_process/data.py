from itertools import chain

import numpy as np
from torch.utils.data import DataLoader, DistributedSampler

import data_process.dataset_factory as dataset_factory
from data_process.clean_factory import EmailCleanConfig, EmailCleaner
from utils.tokenizer_factory import get_caption_tokenizer, get_text_tokenizer

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
    if name != "phish-email":
        raise NotImplementedError(f"Entry dataset {name} not implemented yet.")

    data = dataset_factory.get_dataset(name, mode=mode, cache_dir=cache_dir)

    cleaner = EmailCleaner(
        EmailCleanConfig(
            render_clean_email=True,  # learn clean email format
            mask_urls=True,
            mask_emails=True,
            mask_phones=True,
            truncate_on_thread_markers=True,
            truncate_on_long_quote_block=True,
            strip_common_disclaimers=True,
            drop_if_symbol_ratio_gt=0.60,  # lenient enough for older mailing-list content
            max_body_chars=4000,
        )
    )

    def is_valid_example(example):
        if (
            "text" not in example
            or example["text"] is None
            or len(example["text"].strip()) == 0
            or "style_caption" not in example
            or example["style_caption"] is None
            or len(example["style_caption"].strip()) == 0
        ):
            return False
        cleaned, _ = cleaner.render(example["style_caption"], example["text"])
        return cleaned is not None

    data = data.filter(is_valid_example, num_proc=num_proc)

    tokenizer_text = get_text_tokenizer(text_tokenizer_name)
    eos_id = tokenizer_text.eos_token_id
    pad_id = tokenizer_text.pad_token_id

    tokenizer_caption = get_caption_tokenizer(caption_tokenizer_name)

    def preprocess_and_tokenize(batch):
        texts = batch["text"]
        captions = batch["style_caption"]
        labels = batch.get("phish", [0] * len(texts))

        clean_texts = []
        prefixed_captions = []
        for t, c, p in zip(texts, captions, labels):
            label = int(p) if p is not None else 0
            prefix = (
                "This is a phish email. " if label == 1 else "This is a benign email. "
            )
            raw_caption = (c or "").strip()
            caption = f"{prefix}{raw_caption}".strip()
            cleaned, _ = cleaner.render(raw_caption, t)
            if cleaned is None:
                cleaned = f"{cleaner.cfg.body_prefix}[INVALID SAMPLE]\n"
            clean_texts.append(cleaned)
            prefixed_captions.append(caption)

        enc_text = tokenizer_text(
            clean_texts,
            return_attention_mask=True,
            add_special_tokens=True,
            max_length=text_max_length,
            truncation=True,
            padding="max_length",
        )
        # Keep only one visible EOS: the last token in the visible span.
        for ids, mask in zip(enc_text["input_ids"], enc_text["attention_mask"]):
            seq_len = len(ids)
            visible_len = sum(mask)
            visible_ids = ids[:visible_len]

            content_wo_eos = [t for t in visible_ids if t != eos_id]

            if len(content_wo_eos) >= seq_len:
                new_visible_ids = content_wo_eos[:seq_len]
                new_visible_ids[-1] = eos_id
            else:
                new_visible_ids = content_wo_eos + [eos_id]

            new_visible_len = len(new_visible_ids)
            new_ids = new_visible_ids + [pad_id] * (seq_len - new_visible_len)
            new_mask = [1] * new_visible_len + [0] * (seq_len - new_visible_len)

            ids[:] = new_ids
            mask[:] = new_mask

        # lightweight validation on the first sample in batch
        if enc_text["input_ids"]:
            ids = enc_text["input_ids"][0]
            mask = enc_text["attention_mask"][0]
            visible_tokens = [t for t, m in zip(ids, mask) if m == 1]
            eos_count = sum(1 for t in visible_tokens if t == eos_id)
            if eos_count != 1 or visible_tokens[-1] != eos_id:
                raise ValueError(
                    "Expected exactly one visible EOS and it must be the last visible token."
                )
        enc_cap = tokenizer_caption(
            prefixed_captions,
            return_attention_mask=True,
            add_special_tokens=True,
            max_length=caption_max_length,
            truncation=True,
            padding="max_length",
        )

        return {
            "text_input_ids": enc_text["input_ids"],
            "text_attention_mask": enc_text["attention_mask"],
            "style_caption_input_ids": enc_cap["input_ids"],
            "style_caption_attention_mask": enc_cap["attention_mask"],
        }

    tokenized_dataset = data.map(
        preprocess_and_tokenize,
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=True,
    )

    tokenized_dataset = tokenized_dataset.remove_columns("text")
    tokenized_dataset = tokenized_dataset.remove_columns("style_caption")
    tokenized_dataset = tokenized_dataset.remove_columns("labels")

    tokenized_dataset = tokenized_dataset.with_format("torch")
    return tokenized_dataset


def get_dataloaders(config, distributed=True):
    worker_cfg = config.worker if "worker" in config else config

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

    phase = str(getattr(config, "phase", "")).strip().lower()
    if phase == "pretrain":
        data_format = "chunk"
    elif phase == "finetune":
        data_format = "entry"
    elif phase:
        raise ValueError(
            f"Unknown phase '{config.phase}', expected one of: pretrain, finetune."
        )
    else:
        data_format = config.data.format

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
