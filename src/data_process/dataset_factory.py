import json
from typing import Callable, Dict

from datasets import Dataset, load_dataset


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
