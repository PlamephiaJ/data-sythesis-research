#!/usr/bin/env python3
"""
Clean phishing email dataset using EmailCleaner
Processes all JSON files in data_phish/json/ and exports cleaned data
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from src.data_process.clean_factory import EmailCleanConfig, EmailCleaner


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_json_lines(file_path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file (one JSON object per line)"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Failed to parse line {line_num} in {file_path.name}: {e}"
                )
                continue
    return data


def save_json_lines(data: List[Dict[str, Any]], file_path: Path) -> None:
    """Save data as JSONL file (one JSON object per line)"""
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def clean_dataset(
    input_file: Path,
    output_file: Path,
    cleaner: EmailCleaner,
    caption_field: str = "caption",
) -> Dict[str, Any]:
    """
    Clean a single dataset file

    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        cleaner: EmailCleaner instance
        caption_field: Name of the caption/subject field (if exists)

    Returns:
        Statistics dict
    """
    logger.info(f"Processing {input_file.name}...")

    # Load data
    data = load_json_lines(input_file)
    logger.info(f"  Loaded {len(data)} samples")

    # Clean samples
    cleaned_data = []
    stats = defaultdict(int)
    stats["total_input"] = len(data)

    for idx, sample in enumerate(data):
        if idx % 5000 == 0 and idx > 0:
            logger.info(f"  Processed {idx}/{len(data)} samples...")

        # Extract text and caption
        text = sample.get("text", "")
        caption = sample.get(caption_field, None)

        if not text:
            stats["skipped_no_text"] += 1
            continue

        # Clean the email
        clean_text, drop_reason = cleaner.render(caption, text)

        if drop_reason is not None:
            stats[f"dropped_{drop_reason}"] += 1
            continue

        # Preserve original fields and add cleaned text
        cleaned_sample = sample.copy()
        cleaned_sample["text"] = clean_text
        cleaned_sample["text_original"] = text  # Keep original for reference
        cleaned_sample["cleaning_status"] = "success"

        cleaned_data.append(cleaned_sample)
        stats["successfully_cleaned"] += 1

    # Save cleaned data
    save_json_lines(cleaned_data, output_file)
    stats["total_output"] = len(cleaned_data)
    retention_rate = (len(cleaned_data) / len(data)) * 100 if len(data) > 0 else 0
    stats["retention_rate"] = retention_rate

    logger.info(f"  Saved {len(cleaned_data)} cleaned samples to {output_file.name}")
    logger.info(f"  Retention rate: {retention_rate:.2f}%")

    return dict(stats)


def print_statistics(all_stats: Dict[str, Dict[str, Any]]) -> None:
    """Print cleaning statistics"""
    print("\n" + "=" * 80)
    print("CLEANING STATISTICS")
    print("=" * 80 + "\n")

    total_input = 0
    total_output = 0

    for dataset_name, stats in all_stats.items():
        print(f"\n{dataset_name}:")
        print(f"  Input samples: {stats.get('total_input', 0):,}")
        print(f"  Output samples: {stats.get('total_output', 0):,}")
        print(f"  Retention rate: {stats.get('retention_rate', 0):.2f}%")
        print(f"  Successfully cleaned: {stats.get('successfully_cleaned', 0):,}")

        # Show drop reasons
        drop_reasons = {k: v for k, v in stats.items() if k.startswith("dropped_")}
        if drop_reasons:
            print("  Drop reasons:")
            for reason, count in sorted(drop_reasons.items(), key=lambda x: -x[1]):
                reason_name = reason.replace("dropped_", "")
                pct = (count / stats.get("total_input", 1)) * 100
                print(f"    - {reason_name}: {count:,} ({pct:.2f}%)")

        total_input += stats.get("total_input", 0)
        total_output += stats.get("total_output", 0)

    print("\n" + "-" * 80)
    print(f"TOTAL INPUT: {total_input:,}")
    print(f"TOTAL OUTPUT: {total_output:,}")
    if total_input > 0:
        print(f"OVERALL RETENTION RATE: {(total_output/total_input)*100:.2f}%")
    print("=" * 80 + "\n")


def main():
    # Setup paths
    data_dir = Path(__file__).parent / "data_phish" / "json"
    output_dir = Path(__file__).parent / "data_phish" / "cleaned"

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Initialize cleaner with recommended config for phishing dataset
    config = EmailCleanConfig(
        render_clean_email=True,
        subject_prefix="Subject: ",
        default_subject="Message",
        max_subject_len=90,
        max_body_chars=4000,
        min_body_chars=20,
        truncate_on_thread_markers=True,
        truncate_on_long_quote_block=True,
        long_quote_line_threshold=6,
        strip_common_disclaimers=True,
        disclaimer_tail_scan_chars=1200,
        mask_urls=True,
        mask_emails=True,
        mask_phones=True,
        url_token="<URL>",
        email_token="<EMAIL>",
        phone_token="<PHONE>",
        replace_tabs=True,
        collapse_alignment_spaces=True,
        normalize_separators=True,
        max_blank_lines=2,
        drop_if_contains_replacement_char=True,
        drop_if_symbol_ratio_gt=0.60,
        drop_if_too_short=True,
    )
    cleaner = EmailCleaner(cfg=config)

    logger.info("EmailCleaner initialized with optimized config")
    logger.info(
        f"Config: max_body_chars={config.max_body_chars}, "
        f"min_body_chars={config.min_body_chars}, "
        f"symbol_ratio_threshold={config.drop_if_symbol_ratio_gt}"
    )

    # Process all JSON files
    json_files = sorted(data_dir.glob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files to process")

    all_stats = {}
    for json_file in json_files:
        dataset_name = json_file.stem
        output_file = output_dir / f"{dataset_name}_cleaned.json"

        stats = clean_dataset(json_file, output_file, cleaner)
        all_stats[dataset_name] = stats

    # Print statistics
    print_statistics(all_stats)

    logger.info("Cleaning completed successfully!")


if __name__ == "__main__":
    main()
