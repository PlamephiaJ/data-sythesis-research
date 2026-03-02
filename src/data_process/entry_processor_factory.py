from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from data_process.clean_factory import EmailCleanConfig, EmailCleaner
from utils.tokenizer_factory import get_caption_tokenizer, get_text_tokenizer


@dataclass
class EntryPipeline:
    validate_example: Callable[[Dict[str, Any]], bool]
    preprocess_batch: Callable[[Dict[str, List[Any]]], Dict[str, List[List[int]]]]
    columns_to_remove: List[str]


_REGISTRY: Dict[str, Callable[..., EntryPipeline]] = {}


def register_entry_pipeline(name: str):
    def decorator(factory):
        _REGISTRY[name] = factory
        return factory

    return decorator


def get_entry_pipeline(name: str, **kwargs) -> EntryPipeline:
    if name not in _REGISTRY:
        raise KeyError(f"Entry pipeline '{name}' is not registered")
    return _REGISTRY[name](**kwargs)


def _build_default_email_cleaner() -> EmailCleaner:
    return EmailCleaner(
        EmailCleanConfig(
            render_clean_email=True,
            mask_urls=True,
            mask_emails=True,
            mask_phones=True,
            truncate_on_thread_markers=True,
            truncate_on_long_quote_block=True,
            strip_common_disclaimers=True,
            drop_if_symbol_ratio_gt=0.60,
            max_body_chars=4000,
        )
    )


def _make_email_entry_pipeline(
    text_max_length: int = 1024,
    caption_max_length: int = 256,
    text_tokenizer_name: str = "gpt2",
    caption_tokenizer_name: str = "bert-base-uncased",
) -> EntryPipeline:
    cleaner = _build_default_email_cleaner()
    tokenizer_text = get_text_tokenizer(text_tokenizer_name)
    tokenizer_caption = get_caption_tokenizer(caption_tokenizer_name)

    eos_id = tokenizer_text.eos_token_id
    pad_id = tokenizer_text.pad_token_id

    def validate_example(example: Dict[str, Any]) -> bool:
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

    def preprocess_batch(batch: Dict[str, List[Any]]) -> Dict[str, List[List[int]]]:
        texts = batch["text"]
        captions = batch["style_caption"]
        labels = batch.get("phish", [0] * len(texts))

        clean_texts = []
        prefixed_captions = []
        for text, caption, label in zip(texts, captions, labels):
            numeric_label = int(label) if label is not None else 0
            prefix = (
                "This is a phish email. "
                if numeric_label == 1
                else "This is a benign email. "
            )
            raw_caption = (caption or "").strip()
            formatted_caption = f"{prefix}{raw_caption}".strip()

            cleaned, _ = cleaner.render(raw_caption, text)
            if cleaned is None:
                cleaned = f"{cleaner.cfg.body_prefix}[INVALID SAMPLE]\n"

            clean_texts.append(cleaned)
            prefixed_captions.append(formatted_caption)

        enc_text = tokenizer_text(
            clean_texts,
            return_attention_mask=True,
            add_special_tokens=True,
            max_length=text_max_length,
            truncation=True,
            padding="max_length",
        )

        for ids, mask in zip(enc_text["input_ids"], enc_text["attention_mask"]):
            seq_len = len(ids)
            visible_len = sum(mask)
            visible_ids = ids[:visible_len]

            content_wo_eos = [token for token in visible_ids if token != eos_id]

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

        if enc_text["input_ids"]:
            ids = enc_text["input_ids"][0]
            mask = enc_text["attention_mask"][0]
            visible_tokens = [token for token, marker in zip(ids, mask) if marker == 1]
            eos_count = sum(1 for token in visible_tokens if token == eos_id)
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

    return EntryPipeline(
        validate_example=validate_example,
        preprocess_batch=preprocess_batch,
        columns_to_remove=["text", "style_caption", "labels"],
    )


@register_entry_pipeline("phish-email")
def make_phish_email_entry_pipeline(**kwargs) -> EntryPipeline:
    return _make_email_entry_pipeline(**kwargs)


def _make_cnndailymail_entry_pipeline(
    text_max_length: int = 1024,
    caption_max_length: int = 300,
    text_tokenizer_name: str = "gpt2",
    caption_tokenizer_name: str = "bert-base-uncased",
) -> EntryPipeline:
    """
    Preprocessing pipeline for CNN/DailyMail dataset.

    - article  → text_input_ids (the long-form text to be modeled)
    - highlights → summary_input_ids (the conditioning summary)
    """

    tokenizer_text = get_text_tokenizer(text_tokenizer_name)
    tokenizer_caption = get_caption_tokenizer(caption_tokenizer_name)

    eos_id = tokenizer_text.eos_token_id
    pad_id = tokenizer_text.pad_token_id

    def validate_example(example: Dict[str, Any]) -> bool:
        """Filter out invalid samples: must contain non-empty article and highlights."""
        if (
            "article" not in example
            or example["article"] is None
            or len(str(example["article"]).strip()) == 0
            or "highlights" not in example
            or example["highlights"] is None
            or len(str(example["highlights"]).strip()) == 0
        ):
            return False
        return True

    def preprocess_batch(batch: Dict[str, List[Any]]) -> Dict[str, List[List[int]]]:
        articles = batch["article"]
        summaries = batch["highlights"]

        clean_articles = [(a or "").strip() for a in articles]
        clean_summaries = [(s or "").strip() for s in summaries]

        # ---- Encode article (long text) ----
        enc_text = tokenizer_text(
            clean_articles,
            return_attention_mask=True,
            add_special_tokens=True,
            max_length=text_max_length,
            truncation=True,
            padding="max_length",
        )

        # ---- Normalize EOS positions ----
        for ids, mask in zip(enc_text["input_ids"], enc_text["attention_mask"]):

            seq_len = len(ids)
            visible_len = sum(mask)
            visible_ids = ids[:visible_len]

            # Remove all EOS tokens in visible portion
            content_wo_eos = [tok for tok in visible_ids if tok != eos_id]

            if len(content_wo_eos) >= seq_len:
                # Too long: enforce last token = EOS
                new_visible_ids = content_wo_eos[:seq_len]
                new_visible_ids[-1] = eos_id
            else:
                # Normal case: append exactly one EOS at end
                new_visible_ids = content_wo_eos + [eos_id]

            new_visible_len = len(new_visible_ids)
            new_ids = new_visible_ids + [pad_id] * (seq_len - new_visible_len)
            new_mask = [1] * new_visible_len + [0] * (seq_len - new_visible_len)

            ids[:] = new_ids
            mask[:] = new_mask

        # ---- Sanity check: exactly one visible EOS and it's the last visible token ----
        if enc_text["input_ids"]:
            ids = enc_text["input_ids"][0]
            mask = enc_text["attention_mask"][0]
            visible_tokens = [tok for tok, m in zip(ids, mask) if m == 1]
            eos_count = sum(tok == eos_id for tok in visible_tokens)
            if eos_count != 1 or visible_tokens[-1] != eos_id:
                raise ValueError(
                    "Expected exactly one visible EOS and it must be the last visible token."
                )

        # ---- Encode caption (conditioning) ----
        enc_caption = tokenizer_caption(
            clean_summaries,
            return_attention_mask=True,
            add_special_tokens=True,
            max_length=caption_max_length,
            truncation=True,
            padding="max_length",
        )

        return {
            "text_input_ids": enc_text["input_ids"],
            "text_attention_mask": enc_text["attention_mask"],
            "style_caption_input_ids": enc_caption["input_ids"],
            "style_caption_attention_mask": enc_caption["attention_mask"],
        }

    return EntryPipeline(
        validate_example=validate_example,
        preprocess_batch=preprocess_batch,
        columns_to_remove=["article", "highlights"],
    )


@register_entry_pipeline("abisee/cnn_dailymail")
def make_cnn_dailymail_entry_pipeline(**kwargs) -> EntryPipeline:
    return _make_cnndailymail_entry_pipeline(**kwargs)
