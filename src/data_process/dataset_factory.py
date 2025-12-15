import json
from pathlib import Path
from typing import Callable, Dict

from datasets import Dataset, concatenate_datasets, load_dataset


_REGISTRY: Dict[str, Callable[..., object]] = {}


def register_dataset(name: str):
    """Decorator to register a dataset factory under `name`.

    The decorated object can be a class or a callable factory. The
    registered creator will be called with keyword arguments when
    `get_dataset` is invoked.
    """

    def decorator(obj):
        creator = obj if callable(obj) else None
        if creator is None:
            raise TypeError("Registered object must be callable")
        _REGISTRY[name] = creator
        return obj

    return decorator


def get_dataset(name: str, **kwargs) -> object:
    """Create and return a dataset by `name` using registered creator.

    Example:
        @register_dataset('list')
        def make_list(items):
            return list(items)

        ds = get_dataset('list', items=[1,2,3])
    """

    if name not in _REGISTRY:
        raise KeyError(f"Dataset '{name}' is not registered")
    creator = _REGISTRY[name]
    return creator(**kwargs)


__all__ = ["register_dataset", "get_dataset"]


@register_dataset("wikitext103")
def make_wikitext103(
    mode: str = "train",
    cache_dir: str = None,
    **kwargs,
):
    dataset = load_dataset("wikitext", name="wikitext-103-raw-v1", cache_dir=cache_dir)
    return dataset[mode]


@register_dataset("wikitext2")
def make_wikitext2(
    mode: str = "train",
    cache_dir: str = None,
    **kwargs,
):
    dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", cache_dir=cache_dir)
    return dataset[mode]


@register_dataset("ptb")
def make_ptb(
    mode: str = "train",
    cache_dir: str = None,
    **kwargs,
):
    dataset = load_dataset("ptb_text_only", cache_dir=cache_dir)
    return dataset[mode]


@register_dataset("lambada")
def make_lambada(
    cache_dir: str = None,
    **kwargs,
):
    url = "https://s3.amazonaws.com/research.metamind.io/wikitext/lambada_test.jsonl"

    def read_jsonl_to_list(url):
        import requests

        response = requests.get(url)
        data_list = []
        for line in response.iter_lines():
            if line:
                data_list.append(json.loads(line.decode("utf-8")))
        return data_list

    lambada_data = read_jsonl_to_list(url)
    dataset = Dataset.from_list(lambada_data)
    return dataset


@register_dataset("Plamephia/openwebtext-modern")
def make_openwebtext_modern(
    mode: str = "train",
    cache_dir: str = None,
    **kwargs,
):
    dataset = load_dataset("Plamephia/openwebtext-modern", cache_dir=cache_dir)
    return dataset[mode]


@register_dataset("phish-email")
def make_phish_emails(
    mode: str = "train",
    cache_dir: str = None,
    train_split_ratio: float = 0.9,
    seed: int = 42,
    **kwargs,
):
    dataroot = Path(cache_dir)
    if not dataroot.exists():
        raise FileNotFoundError(f"Data directory {dataroot} does not exist.")

    # Collect all JSON file paths (they are in JSONL format)
    json_files = sorted(dataroot.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {dataroot}")

    # Load and standardize all datasets
    datasets = []
    for json_file in json_files:
        ds = load_dataset("json", data_files=str(json_file))["train"]

        # Convert phish field to int64 for consistency, handle None/float values
        def convert_phish_to_int(ex):
            val = ex.get("phish")
            ex["phish"] = int(val) if val is not None else 0
            return ex

        ds = ds.map(convert_phish_to_int)
        # Cast phish to int64 type
        from datasets.features import Features, Value

        ds = ds.cast(
            Features(
                {
                    k: v if k != "phish" else Value("int64")
                    for k, v in ds.features.items()
                }
            )
        )
        datasets.append(ds)

    # Merge all datasets into one
    merged_dataset = concatenate_datasets(datasets)

    # Split dataset according to mode
    ds_phish = merged_dataset.filter(lambda x: x["phish"] == 1)
    ds_benign = merged_dataset.filter(lambda x: x["phish"] == 0)

    split_phish = ds_phish.train_test_split(test_size=1 - train_split_ratio, seed=seed)
    split_benign = ds_benign.train_test_split(
        test_size=1 - train_split_ratio, seed=seed
    )
    if mode == "train":
        merged_dataset = concatenate_datasets(
            [split_phish["train"], split_benign["train"]]
        )
    elif mode == "validation":
        merged_dataset = concatenate_datasets(
            [split_phish["test"], split_benign["test"]]
        )
    else:
        raise ValueError(f"Unknown mode {mode} for phish-email dataset.")
    return merged_dataset
