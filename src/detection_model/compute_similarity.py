import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_AUGMENTED_PATH = "src/detection_model/augmented_from_caption_all.jsonl"
DEFAULT_REFERENCE_PATH = (
    "data_phish/masked/hdbscan_clustering/phish_clustered_all.jsonl"
)
DEFAULT_OUTPUT_PATH = "src/detection_model/similarity_report.json"


def _iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _extract_text(record: Dict, keys: List[str]) -> Optional[str]:
    for key in keys:
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return None


def load_augmented_samples(path: Path) -> List[str]:
    samples: List[str] = []
    for obj in _iter_jsonl(path):
        text = _extract_text(obj, ["sample"])
        if text:
            samples.append(text)
    return samples


def load_reference_cluster_zero_samples(path: Path) -> List[str]:
    samples: List[str] = []
    for obj in _iter_jsonl(path):
        if int(obj.get("cluster_id", -1)) != 0:
            continue
        text = _extract_text(obj, ["sample", "text", "text_original"])
        if text:
            samples.append(text)
    return samples


def _safe_mean_cosine_from_tfidf(augmented: List[str], reference: List[str]) -> float:
    vectorizer = TfidfVectorizer(
        max_features=50000,
        lowercase=True,
        stop_words="english",
        token_pattern=r"(?u)\b\w\w+\b",
    )
    all_docs = augmented + reference
    x = vectorizer.fit_transform(all_docs)
    n_aug = len(augmented)
    aug_mean = np.asarray(x[:n_aug].mean(axis=0))
    ref_mean = np.asarray(x[n_aug:].mean(axis=0))
    score = cosine_similarity(aug_mean, ref_mean)[0, 0]
    return float(score)


def _token_distribution_metrics(
    augmented: List[str], reference: List[str]
) -> Dict[str, float]:
    vectorizer = CountVectorizer(
        max_features=50000,
        lowercase=True,
        stop_words="english",
        token_pattern=r"(?u)\b\w\w+\b",
    )
    all_docs = augmented + reference
    x = vectorizer.fit_transform(all_docs)
    n_aug = len(augmented)

    aug_counts = np.asarray(x[:n_aug].sum(axis=0)).ravel().astype(np.float64)
    ref_counts = np.asarray(x[n_aug:].sum(axis=0)).ravel().astype(np.float64)

    eps = 1e-12
    p = (aug_counts + eps) / (aug_counts.sum() + eps * len(aug_counts))
    q = (ref_counts + eps) / (ref_counts.sum() + eps * len(ref_counts))

    js_distance = float(jensenshannon(p, q, base=2.0))
    js_similarity = 1.0 - js_distance

    feature_names = vectorizer.get_feature_names_out()
    top_k = min(2000, len(feature_names))
    top_aug_ids = np.argpartition(aug_counts, -top_k)[-top_k:]
    top_ref_ids = np.argpartition(ref_counts, -top_k)[-top_k:]
    set_aug = {feature_names[i] for i in top_aug_ids}
    set_ref = {feature_names[i] for i in top_ref_ids}
    jaccard = len(set_aug & set_ref) / max(1, len(set_aug | set_ref))

    return {
        "js_distance": js_distance,
        "js_similarity": float(js_similarity),
        "top_token_jaccard": float(jaccard),
        "vocab_size": float(len(feature_names)),
    }


def _length_metrics(augmented: List[str], reference: List[str]) -> Dict[str, float]:
    aug_lengths = np.array([len(x) for x in augmented], dtype=np.float64)
    ref_lengths = np.array([len(x) for x in reference], dtype=np.float64)

    wd = float(wasserstein_distance(aug_lengths, ref_lengths))
    length_similarity = 1.0 / (1.0 + wd)

    return {
        "aug_mean_length": float(aug_lengths.mean()),
        "ref_mean_length": float(ref_lengths.mean()),
        "aug_std_length": float(aug_lengths.std()),
        "ref_std_length": float(ref_lengths.std()),
        "length_wasserstein": wd,
        "length_similarity": float(length_similarity),
    }


def compute_distribution_similarity(augmented: List[str], reference: List[str]) -> Dict:
    if not augmented:
        raise ValueError("Augmented samples are empty.")
    if not reference:
        raise ValueError("Reference samples with cluster_id==0 are empty.")

    tfidf_cosine = _safe_mean_cosine_from_tfidf(augmented, reference)
    token_metrics = _token_distribution_metrics(augmented, reference)
    length_metrics = _length_metrics(augmented, reference)

    overall_score = float(
        np.mean(
            [
                tfidf_cosine,
                token_metrics["js_similarity"],
                token_metrics["top_token_jaccard"],
                length_metrics["length_similarity"],
            ]
        )
    )

    return {
        "sample_count": {
            "augmented": len(augmented),
            "reference_cluster_0": len(reference),
        },
        "similarity": {
            "tfidf_mean_cosine": tfidf_cosine,
            **token_metrics,
            **length_metrics,
            "overall_similarity": overall_score,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare distribution similarity between augmented samples and cluster_id==0 reference samples."
    )
    parser.add_argument(
        "--augmented_path",
        type=str,
        default=DEFAULT_AUGMENTED_PATH,
        help="Path to augmented jsonl containing `sample` field.",
    )
    parser.add_argument(
        "--reference_path",
        type=str,
        default=DEFAULT_REFERENCE_PATH,
        help="Path to clustered reference jsonl containing `cluster_id` and text fields.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save similarity report json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    augmented_path = Path(args.augmented_path)
    reference_path = Path(args.reference_path)
    output_path = Path(args.output_path)

    if not augmented_path.exists():
        raise FileNotFoundError(f"Augmented file not found: {augmented_path}")
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference file not found: {reference_path}")

    augmented_samples = load_augmented_samples(augmented_path)
    reference_samples = load_reference_cluster_zero_samples(reference_path)

    result = compute_distribution_similarity(augmented_samples, reference_samples)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Augmented samples: {result['sample_count']['augmented']}")
    print(
        f"Reference samples (cluster_id==0): {result['sample_count']['reference_cluster_0']}"
    )
    print("=== Similarity Metrics ===")
    for k, v in result["similarity"].items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")
    print(f"Saved report to: {output_path}")


if __name__ == "__main__":
    main()
