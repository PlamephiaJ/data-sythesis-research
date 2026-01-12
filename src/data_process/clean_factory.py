# data_process/email_cleaner.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple


# ---------------------------
# Config
# ---------------------------


@dataclass(frozen=True)
class EmailCleanConfig:
    # Render mode
    render_clean_email: bool = True  # if True -> Subject + blank line + body
    subject_prefix: str = "Subject: "
    body_prefix: str = "Body: "
    default_subject: str = "Message"
    max_subject_len: int = 90

    # Length controls
    max_body_chars: int = 4000
    min_body_chars: int = 20

    # Thread/quote handling
    truncate_on_thread_markers: bool = True
    truncate_on_long_quote_block: bool = True
    long_quote_line_threshold: int = (
        6  # if >= N quoted lines, truncate at first quote line
    )

    # Footer/disclaimer handling
    strip_common_disclaimers: bool = True
    disclaimer_tail_scan_chars: int = 1200

    # PII / spam artifact masking
    mask_urls: bool = True
    mask_emails: bool = True
    mask_phones: bool = True
    url_token: str = "<URL>"
    email_token: str = "<EMAIL>"
    phone_token: str = "<PHONE>"

    # Formatting cleanup
    replace_tabs: bool = True
    collapse_alignment_spaces: bool = True  # collapse 3+ spaces used for ASCII layout
    normalize_separators: bool = True  # normalize long separator lines to '---'
    max_blank_lines: int = 2

    # Quality gates (you said "不隔离" but you still need to prevent catastrophic noise)
    # Keep policy: return cleaned text; if unrecoverable -> mark invalid.
    drop_if_contains_replacement_char: bool = True  # '�'
    drop_if_symbol_ratio_gt: float = 0.60
    drop_if_too_short: bool = True


# ---------------------------
# Regexes (module-level => picklable)
# ---------------------------

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE_RE = re.compile(r"\b(\+?\d[\d\-\s().]{7,}\d)\b")

_THREAD_MARKERS = [
    re.compile(r"^-+\s*Original Message\s*-+$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^On .+wrote:\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^-----\s*Forwarded message\s*-----", re.IGNORECASE | re.MULTILINE),
    # common "From/Sent/To/Subject" block inside body
    re.compile(
        r"^From:\s.+\n(Sent:|Date:)\s.+\n(To:|Cc:)\s.+", re.IGNORECASE | re.MULTILINE
    ),
]
_QUOTE_LINE_RE = re.compile(r"^\s*>", re.MULTILINE)

_SEPARATOR_LINE_RE = re.compile(r"^[=\-_*]{5,}\s*$", re.MULTILINE)
_MULTI_SPACE_RE = re.compile(r"[ ]{3,}")
_BLANKLINES_RE = re.compile(r"\n{3,}")

_DISCLAIMER_HINTS = (
    "confidential",
    "privileged",
    "intended recipient",
    "unauthorized",
    "do not distribute",
    "disclaimer",
    "liability",
    "virus",
    "monitoring",
)


# ---------------------------
# Utilities
# ---------------------------


def _symbol_ratio(s: str) -> float:
    if not s:
        return 1.0
    normal = 0
    for ch in s:
        if ch.isalnum() or ch.isspace() or ch in ".,;:!?'-\"()[]/\\@<>":
            normal += 1
    return 1.0 - (normal / len(s))


def _clip(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[:n].rstrip() + "\n\n[...truncated...]"


def _normalize_newlines(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")


# ---------------------------
# Pipeline step protocol (Strategy)
# ---------------------------


class Step(Protocol):

    def apply(self, text: str, cfg: EmailCleanConfig) -> Tuple[str, Optional[str]]:
        """
        Returns: (text, drop_reason)
        drop_reason is None if keep.
        """
        ...


# ---------------------------
# Steps
# ---------------------------


class QualityGate:

    def apply(self, text: str, cfg: EmailCleanConfig) -> Tuple[str, Optional[str]]:
        if cfg.drop_if_contains_replacement_char and "�" in text:
            return text, "contains_replacement_char"
        if cfg.drop_if_too_short and len(text.strip()) < cfg.min_body_chars:
            return text, "too_short"
        if _symbol_ratio(text) > cfg.drop_if_symbol_ratio_gt:
            return text, "symbol_ratio_too_high"
        return text, None


class TruncateThreadsAndQuotes:

    def apply(self, text: str, cfg: EmailCleanConfig) -> Tuple[str, Optional[str]]:
        cut_positions: List[int] = []

        if cfg.truncate_on_thread_markers:
            for pat in _THREAD_MARKERS:
                m = pat.search(text)
                if m:
                    cut_positions.append(m.start())

        if cfg.truncate_on_long_quote_block:
            qcount = len(_QUOTE_LINE_RE.findall(text))
            if qcount >= cfg.long_quote_line_threshold:
                qm = _QUOTE_LINE_RE.search(text)
                if qm:
                    cut_positions.append(qm.start())

        if cut_positions:
            text = text[: min(cut_positions)].rstrip()
        return text, None


class StripDisclaimerTail:

    def apply(self, text: str, cfg: EmailCleanConfig) -> Tuple[str, Optional[str]]:
        if not cfg.strip_common_disclaimers or not text:
            return text, None

        tail = text[-cfg.disclaimer_tail_scan_chars :].lower()
        if any(h in tail for h in _DISCLAIMER_HINTS):
            seps = list(_SEPARATOR_LINE_RE.finditer(text))
            if seps:
                last = seps[-1]
                if last.start() >= max(0, len(text) - cfg.disclaimer_tail_scan_chars):
                    text = text[: last.start()].rstrip()
        return text, None


class MaskArtifacts:

    def apply(self, text: str, cfg: EmailCleanConfig) -> Tuple[str, Optional[str]]:
        if cfg.mask_urls:
            text = _URL_RE.sub(cfg.url_token, text)
        if cfg.mask_emails:
            text = _EMAIL_RE.sub(cfg.email_token, text)
        if cfg.mask_phones:
            text = _PHONE_RE.sub(cfg.phone_token, text)
        return text, None


class NormalizeFormatting:

    def apply(self, text: str, cfg: EmailCleanConfig) -> Tuple[str, Optional[str]]:
        if cfg.replace_tabs:
            text = text.replace("\t", " ")

        if cfg.normalize_separators:
            text = _SEPARATOR_LINE_RE.sub("---", text)

        if cfg.collapse_alignment_spaces:
            text = _MULTI_SPACE_RE.sub("  ", text)

        # cap blank lines
        text = _BLANKLINES_RE.sub("\n\n", text)

        text = text.strip()
        return text, None


class ClipLength:

    def apply(self, text: str, cfg: EmailCleanConfig) -> Tuple[str, Optional[str]]:
        return _clip(text, cfg.max_body_chars), None


# ---------------------------
# Cleaner (Pipeline / Chain of Responsibility)
# ---------------------------


class EmailCleaner:
    """
    Pipeline that turns raw email body -> cleaned body, and optionally renders
    to a stable email format. Designed to be used inside datasets.map.
    """

    def __init__(
        self, cfg: Optional[EmailCleanConfig] = None, steps: Optional[List[Step]] = None
    ):
        self.cfg = cfg or EmailCleanConfig()
        self.steps = steps or [
            QualityGate(),
            TruncateThreadsAndQuotes(),
            StripDisclaimerTail(),
            MaskArtifacts(),
            NormalizeFormatting(),
            ClipLength(),
        ]

    def clean_body(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        text = _normalize_newlines(text or "").strip()
        for step in self.steps:
            text, reason = step.apply(text, self.cfg)
            if reason is not None:
                return None, reason
        if not text.strip():
            return None, "empty_after_clean"
        return text, None

    def derive_subject(self, caption: Optional[str], body: str) -> str:
        cfg = self.cfg
        subject = (caption or "").strip()

        if not subject:
            first_line = body.split("\n", 1)[0].strip()
            if 3 <= len(first_line) <= 80 and not re.match(
                r"^(hi|hello|greetings)\b", first_line, re.I
            ):
                subject = first_line
            else:
                subject = cfg.default_subject

        subject = subject.replace("\n", " ")
        subject = re.sub(r"\s{2,}", " ", subject).strip()
        if len(subject) > cfg.max_subject_len:
            subject = subject[: cfg.max_subject_len - 1].rstrip() + "…"
        return subject

    def render(
        self, caption: Optional[str], raw_text: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Returns (clean_email_or_body, reason)
        """
        body, reason = self.clean_body(raw_text)
        if reason is not None:
            return None, reason

        if not self.cfg.render_clean_email:
            return body, None

        subject = self.derive_subject(caption, body)
        return (
            f"{self.cfg.subject_prefix}{subject}\n\n{self.cfg.body_prefix}{body}\n",
            None,
        )
