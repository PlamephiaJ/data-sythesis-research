from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence

import numpy as np


# =========================
# Utility: basic text split
# =========================


def split_paragraphs(text: str) -> List[str]:
    """
    Split text into paragraph-like chunks.
    - Splits on blank lines first.
    - Falls back to sentence-ish splitting if no blank lines.
    """
    text = (text or "").strip()
    if not text:
        return []

    # Paragraph split on blank lines
    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if len(paras) >= 2:
        return paras

    # Fallback: split on sentence boundaries (very simple heuristic)
    # Keep it conservative to avoid over-splitting emails.
    sents = [s.strip() for s in re.split(r"(?<=[.!?。！？])\s+", text) if s.strip()]
    if len(sents) >= 2:
        # Group into ~2-4 sentences per chunk to reduce noise
        chunks: List[str] = []
        buf: List[str] = []
        for s in sents:
            buf.append(s)
            if len(buf) >= 3:
                chunks.append(" ".join(buf))
                buf = []
        if buf:
            chunks.append(" ".join(buf))
        return chunks

    return [text]


def l2_normalize(mat: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(mat, axis=axis, keepdims=True)
    return mat / np.maximum(norm, eps)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a: (N, D), b: (M, D) -> (N, M)
    """
    a_n = l2_normalize(a)
    b_n = l2_normalize(b)
    return a_n @ b_n.T


# =========================
# Strategy: Embedding backend
# =========================


class EmbeddingBackend(Protocol):

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return embeddings as float32 ndarray of shape (len(texts), dim)."""


@dataclass(frozen=True)
class SentenceTransformersBackend:
    """
    Strategy implementation: sentence-transformers.

    Requires:
      uv add sentence-transformers
    """

    model_name: str
    device: Optional[str] = None  # e.g., "cpu", "cuda"

    def __post_init__(self):
        from sentence_transformers import SentenceTransformer

        object.__setattr__(
            self, "_model", SentenceTransformer(self.model_name, device=self.device)
        )

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        emb = self._model.encode(
            list(texts),
            normalize_embeddings=True,  # important for cosine stability
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        return emb


@dataclass(frozen=True)
class TransformersMeanPoolingBackend:
    """
    Strategy implementation: HuggingFace Transformers with mean pooling.
    Works for general encoders, but you should pick a model suitable for embeddings.
    """

    model_name: str
    device: Optional[str] = None  # e.g., "cpu", "cuda"
    max_length: int = 512
    batch_size: int = 32

    def __post_init__(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        model = AutoModel.from_pretrained(self.model_name)
        model.eval()

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(self.device)

        object.__setattr__(self, "_torch", torch)
        object.__setattr__(self, "_tokenizer", tokenizer)
        object.__setattr__(self, "_model", model)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)

        torch = self._torch
        tokenizer = self._tokenizer
        model = self._model

        all_embs: List[np.ndarray] = []
        texts_list = list(texts)

        with torch.no_grad():
            for i in range(0, len(texts_list), self.batch_size):
                batch = texts_list[i : i + self.batch_size]
                enc = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                out = model(**enc)

                # Mean pooling over token embeddings with attention mask
                token_emb = out.last_hidden_state  # (B, L, H)
                mask = (
                    enc["attention_mask"].unsqueeze(-1).expand(token_emb.size()).float()
                )
                sum_emb = (token_emb * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp(min=1e-9)
                sent_emb = sum_emb / denom  # (B, H)

                sent_emb = sent_emb.cpu().numpy().astype(np.float32)
                sent_emb = l2_normalize(sent_emb)
                all_embs.append(sent_emb)

        return np.vstack(all_embs)


# =========================
# Policy: aggregation strategy (long email)
# =========================


class AggregationPolicy(Protocol):

    def aggregate(self, caption_emb: np.ndarray, chunk_embs: np.ndarray) -> float:
        """
        caption_emb: (1, D)
        chunk_embs: (K, D)
        Returns scalar alignment score.
        """


@dataclass(frozen=True)
class WholeDocPolicy:
    """
    Aggregate by embedding the whole email (single chunk expected).
    If multiple chunks are provided, averages them first.
    """

    def aggregate(self, caption_emb: np.ndarray, chunk_embs: np.ndarray) -> float:
        if chunk_embs.shape[0] == 0:
            return float("nan")
        doc = chunk_embs.mean(axis=0, keepdims=True)
        return float(cosine_sim(caption_emb, doc)[0, 0])


@dataclass(frozen=True)
class MaxSimPolicy:
    """
    Aggregate by maximum similarity across chunks:
    detects whether any paragraph strongly aligns with the caption.
    """

    def aggregate(self, caption_emb: np.ndarray, chunk_embs: np.ndarray) -> float:
        if chunk_embs.shape[0] == 0:
            return float("nan")
        sims = cosine_sim(caption_emb, chunk_embs)[0]  # (K,)
        return float(np.max(sims))


@dataclass(frozen=True)
class TopKMeanPolicy:
    """
    Aggregate by mean of top-k similarities (more stable than MaxSim).
    """

    k: int = 3

    def aggregate(self, caption_emb: np.ndarray, chunk_embs: np.ndarray) -> float:
        if chunk_embs.shape[0] == 0:
            return float("nan")
        sims = cosine_sim(caption_emb, chunk_embs)[0]  # (K,)
        k = min(self.k, sims.shape[0])
        topk = np.partition(sims, -k)[-k:]
        return float(np.mean(topk))


# =========================
# Facade: Sentence-Embedding Alignment metric
# =========================


@dataclass
class SentenceEmbeddingAlignment:
    """
    Facade that ties together:
    - EmbeddingBackend (Strategy)
    - AggregationPolicy (Policy)
    - Chunking strategy (here: paragraphs/sentence groups)

    Typical usage:
      metric = SentenceEmbeddingAlignment(
          backend=SentenceTransformersBackend("intfloat/e5-base-v2"),
          policy=TopKMeanPolicy(k=3),
          chunker=split_paragraphs,
          prefix_caption="query: ",
          prefix_chunk="passage: ",
      )
      score = metric.score(caption, email)
    """

    backend: EmbeddingBackend
    policy: AggregationPolicy
    chunker: callable = split_paragraphs

    # Optional prefixes (useful for E5-style models that expect "query:"/"passage:")
    prefix_caption: str = ""
    prefix_chunk: str = ""

    def score(self, caption: str, email: str) -> float:
        caption = (caption or "").strip()
        email = (email or "").strip()
        if not caption or not email:
            return float("nan")

        chunks = self.chunker(email)
        if not chunks:
            return float("nan")

        cap_text = self.prefix_caption + caption
        chunk_texts = [self.prefix_chunk + c for c in chunks]

        cap_emb = self.backend.embed([cap_text])  # (1, D)
        chunk_embs = self.backend.embed(chunk_texts)  # (K, D)
        return self.policy.aggregate(cap_emb, chunk_embs)

    def score_batch(self, captions: Sequence[str], emails: Sequence[str]) -> np.ndarray:
        """
        Batch scoring.
        Note: because chunking yields variable number of chunks, we do per-example aggregation.
        For high throughput, you can pre-chunk + flatten then embed once; keep this MVP first.
        """
        if len(captions) != len(emails):
            raise ValueError("captions and emails must have the same length")

        scores: List[float] = []
        for c, e in zip(captions, emails):
            scores.append(self.score(c, e))
        return np.array(scores, dtype=np.float32)


# =========================
# Factory: recommended presets
# =========================


def make_default_alignment_metric(
    model_name: str = "intfloat/e5-base-v2",
    use_sentence_transformers: bool = True,
    device: Optional[str] = None,
    policy: Optional[AggregationPolicy] = None,
) -> SentenceEmbeddingAlignment:
    """
    Sensible defaults for caption (short) -> email (long).
    - E5 model is a strong default for query-vs-passage style matching.
    - TopKMeanPolicy is more stable than MaxSim for production monitoring.
    """
    if policy is None:
        policy = TopKMeanPolicy(k=3)

    if use_sentence_transformers:
        backend = SentenceTransformersBackend(model_name=model_name, device=device)
    else:
        backend = TransformersMeanPoolingBackend(model_name=model_name, device=device)

    # E5 convention: "query:" and "passage:"
    prefix_caption = "query: " if "e5" in model_name.lower() else ""
    prefix_chunk = "passage: " if "e5" in model_name.lower() else ""

    return SentenceEmbeddingAlignment(
        backend=backend,
        policy=policy,
        prefix_caption=prefix_caption,
        prefix_chunk=prefix_chunk,
    )


if __name__ == "__main__":
    metric = make_default_alignment_metric(
        model_name="intfloat/e5-base-v2",
        use_sentence_transformers=True,
        policy=MaxSimPolicy(),  # or TopKMeanPolicy(k=3)
    )

    caption = "Follow up with the client about the revised quote and ask for approval by Friday."
    email = """Hi Alex,

    Following up on the revised quote we sent yesterday. Could you please confirm approval by Friday so we can proceed?

    Best regards,
    Sam
    """

    print(metric.score(caption, email))
