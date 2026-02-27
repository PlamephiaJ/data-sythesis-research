import argparse
import json
from pathlib import Path


def _parse_cluster_id(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def export_captions_for_cluster(
    input_path: Path,
    cluster_id: int,
    output_path: Path,
    deduplicate: bool = True,
) -> int:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    captions = []
    seen = set()

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            record_cluster_id = _parse_cluster_id(record.get("cluster_id"))
            if record_cluster_id != cluster_id:
                continue

            caption = (record.get("style_caption") or "").strip()
            if not caption:
                continue

            if deduplicate:
                if caption in seen:
                    continue
                seen.add(caption)
            captions.append(caption)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for caption in captions:
            f.write(caption + "\n")

    return len(captions)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract style_caption lines for a specified cluster_id from "
            "phish_clustered_all.jsonl and save as one line per caption."
        )
    )
    parser.add_argument("--cluster_id", type=int, required=True)
    parser.add_argument(
        "--input_path",
        type=str,
        default="data_phish/masked/hdbscan_clustering/phish_clustered_all.jsonl",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help=(
            "Optional output path. Default: src/detection_model/"
            "caption_cluster_id_<cluster_id>.txt"
        ),
    )
    parser.add_argument(
        "--keep_duplicates",
        action="store_true",
        help="Keep duplicated captions. Default behavior is deduplicate.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = Path(
            f"src/detection_model/caption_cluster_id_{args.cluster_id}.txt"
        )

    count = export_captions_for_cluster(
        input_path=input_path,
        cluster_id=args.cluster_id,
        output_path=output_path,
        deduplicate=not args.keep_duplicates,
    )

    print(f"Saved {count} caption(s) to: {output_path}")


if __name__ == "__main__":
    main()
