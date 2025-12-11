import re


# Minimal registry for detokenizers
_REGISTRY = {}


def register_detokenizer(name: str):
    """Decorator to register a detokenizer under `name`.

    Example:
        @register_detokenizer('wikitext')
        def wt_detokenizer(s): ...
    """

    def dec(fn):
        if not callable(fn):
            raise TypeError("detokenizer must be callable")
        _REGISTRY[name] = fn
        return fn

    return dec


def get_detokenizer(name: str):
    """Return a registered detokenizer by name.

    Behavior:
    - If `name` exactly matches a registered detokenizer key, return it.
    - Otherwise, treat `name` as a dataset name and try to map it using
      `choose_for_dataset` (e.g. 'wikitext-103-raw-v1' -> 'wikitext').

    Raises KeyError if no detokenizer can be found/mapped.
    """

    # direct registered key
    if name in _REGISTRY:
        return _REGISTRY[name]

    # try to interpret name as a dataset identifier and map it
    try:
        return choose_for_dataset(name)
    except KeyError:
        raise KeyError(
            f"Detokenizer '{name}' is not registered and no mapping available"
        )


def choose_for_dataset(dataset_name: str):
    """Choose an appropriate detokenizer for a dataset name.

    Mirrors the previous ad-hoc logic: dataset names starting with
    "wikitext" use the wikitext detokenizer; others match exact
    names like 'ptb', 'lm1b', 'lambada'. Raises KeyError if none.
    """

    if dataset_name.startswith("wikitext"):
        return get_detokenizer("wikitext")
    if dataset_name == "ptb":
        return get_detokenizer("ptb")
    if dataset_name == "lm1b":
        return get_detokenizer("lm1b")
    if dataset_name == "lambada":
        return get_detokenizer("lambada")
    raise KeyError(f"No detokenizer known for dataset '{dataset_name}'")


@register_detokenizer("wikitext")
def wt_detokenizer(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string


@register_detokenizer("ptb")
def ptb_detokenizer(x):
    x = x.replace(" 's", "'s")
    x = x.replace("s ' ", "s' ")
    x = x.replace(" n't", "n't")
    x = x.replace(" \n ", "\n")
    x = x.replace("\\/", "/")
    for _ in range(10):
        x = x.replace(" N ", " 1 ")
    x = x.replace("$ 1", "$1")
    x = x.replace("# 1", "#1")
    x = x.replace("<unk>", "?")
    return x


@register_detokenizer("lm1b")
def lm1b_detokenizer(x):
    x = x.replace("http : / / ", "http://")
    x = x.replace("https : / / ", "https://")
    x = re.sub(r" \'(\w+)", r"'\1", x)
    x = re.sub(r" (\w+) \. ", r" \1. ", x)
    x = re.sub(r" (\w+) \.$", r" \1.", x)
    x = x.replace(" ? ", "? ")
    x = re.sub(r" \?$", "?", x)
    x = x.replace(" ! ", "! ")
    x = re.sub(r" \!$", "!", x)
    x = x.replace(" , ", ", ")
    x = x.replace(" : ", ": ")
    x = x.replace(" ; ", "; ")
    x = x.replace(" / ", "/")
    x = re.sub(r"\" ([^\"]+) \"", r'"\1"', x)
    x = re.sub(r"\' ([^\']+) \'", r"'\1'", x)
    x = re.sub(r"\( ([^\(\)]+) \)", r"(\1)", x)
    x = re.sub(r"\[ ([^\[\]]+) \]", r"[\1]", x)
    x = x.replace("$ ", "$")
    x = x.replace("£ ", "£")
    return x


@register_detokenizer("lambada")
def lambada_detokenizer(text):
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    return "\n" + text.strip()


__all__ = [
    "register_detokenizer",
    "get_detokenizer",
    "choose_for_dataset",
    "wt_detokenizer",
    "ptb_detokenizer",
    "lm1b_detokenizer",
    "lambada_detokenizer",
]
