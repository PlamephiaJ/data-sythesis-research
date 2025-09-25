"""Data module utilities for building loaders used during training and evaluation."""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader, DistributedSampler

from .preprocess import get_dataset


def cycle_loader(dataloader: DataLoader, sampler: Optional[DistributedSampler] = None) -> Iterator:
    """Yield data from a loader indefinitely, resetting distributed samplers each cycle."""
    while True:
        if sampler is not None:
            sampler.set_epoch(np.random.randint(0, 100_000))
        for batch in dataloader:
            yield batch


def get_dataloaders(config) -> Tuple[Iterator, Iterator]:
    """Construct distributed-aware iterators for train and validation datasets."""
    if config.training.batch_size % (config.ngpus * config.training.accum) != 0:
        raise ValueError(
            "Train Batch Size {} is not divisible by {} gpus with accumulation {}.".format(
                config.training.batch_size, config.ngpus, config.training.accum
            )
        )
    if config.eval.batch_size % (config.ngpus * config.training.accum) != 0:
        raise ValueError(
            "Eval Batch Size {} is not divisible by {} gpus with accumulation {}.".format(
                config.eval.batch_size, config.ngpus, config.training.accum
            )
        )

    train_set = get_dataset(
        config.data.train, "train", cache_dir=config.data.cache_dir, block_size=config.model.length
    )
    valid_set = get_dataset(
        config.data.valid,
        "validation" if config.data.valid != "text8" else "test",
        cache_dir=config.data.cache_dir,
        block_size=config.model.length,
    )

    if config.ngpus > 1:
        train_sampler: Optional[DistributedSampler] = DistributedSampler(train_set)
        valid_sampler: Optional[DistributedSampler] = DistributedSampler(valid_set)
    else:
        train_sampler = None
        valid_sampler = None

    per_device_batch = config.training.batch_size // (config.ngpus * config.training.accum)
    train_loader = cycle_loader(
        DataLoader(
            train_set,
            batch_size=per_device_batch,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
            shuffle=(train_sampler is None),
            persistent_workers=True,
        ),
        sampler=train_sampler,
    )

    eval_loader = cycle_loader(
        DataLoader(
            valid_set,
            batch_size=config.eval.batch_size // (config.ngpus * config.training.accum),
            sampler=valid_sampler,
            num_workers=4,
            pin_memory=True,
            shuffle=(valid_sampler is None),
        ),
        sampler=valid_sampler,
    )

    return train_loader, eval_loader
