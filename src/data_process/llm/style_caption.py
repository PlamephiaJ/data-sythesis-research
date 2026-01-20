"""The script to run the Ollama LLM model for style captioning.
The Ollama is very slow, future work should consider using a faster framework like VLLM.
"""

import json
import logging
from pathlib import Path
from typing import List

import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DATA_DIRECTORY = Path("data_phish") / "raw"

MODEL = "qwen3:32b"
LLM = OllamaLLM(model=MODEL)


def llm_caption_style(email_text: str) -> str:
    """Generate a style caption based on the given email using the Ollama LLM model."""
    template = (
        "Given the following email, generate a concise style caption that captures its tone and intent. "
        "You should describe the email's content and purpose concisely and accurately. Do not include any greetings or sign-offs. "
        "Respond quickly and do not use chain of thought. Your answer should be placed in between <caption> </caption>\n\n"
        "Raw Email:\n\n{email}\n\n"
    )
    prompt = PromptTemplate.from_template(template)
    full_prompt = prompt.format(email=email_text)
    try:
        response = LLM.invoke(full_prompt)
        caption = response.split("<caption>")[1].split("</caption>")[0].strip()
    except Exception as e:
        raise RuntimeError(
            f"Error generating caption: {e}, input: {full_prompt}, response: {response}"
        )
    return caption


def process_email_dataset(dataset_json: Path) -> List:
    """Process the email dataset to generate style captions for each email. Returns a list of the failed extraction ids."""
    df = pd.read_json(dataset_json, lines=True)
    failed_ids = []
    max_retries = 5
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing emails"):
        email_text = row.get("text", "")
        email_id = row.get("id", idx)
        retries = 0
        while retries < max_retries:
            try:
                caption = llm_caption_style(email_text)
                df.at[idx, "style_caption"] = caption
            except Exception as e:
                logger.error(f"Failed to process email id {email_id}: {e}")
            retries += 1
            if retries == max_retries:
                failed_ids.append(email_id)
    output_file = dataset_json.with_name(f"{dataset_json.stem}_with_captions.json")
    output_file = Path("data_phish") / "jsonl" / output_file.name
    df.to_json(output_file, orient="records", lines=True)
    logger.info(f"Processed dataset saved to {output_file}")
    return failed_ids


if __name__ == "__main__":
    data_dir = RAW_DATA_DIRECTORY.absolute()
    logger.info("Using Ollama model: %s", MODEL)
    logger.info("Raw data directory: %s", data_dir)
    failed_ids_dict = {}
    for dataset_file in data_dir.glob("*.json"):
        logger.info("Processing dataset file: %s", dataset_file)
        failed_ids = process_email_dataset(dataset_file)
        failed_ids_dict[dataset_file.name] = failed_ids
    if any(failed_ids_dict.values()):
        with open(Path("data_phish") / "jsonl" / "failed_ids.json", "w") as f:
            json.dump(failed_ids_dict, f, indent=2)
        logger.info(
            f"Failed IDs saved to {Path('data_phish') / 'jsonl' / 'failed_ids.json'}"
        )
