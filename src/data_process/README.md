# Data Processing Workspace

The `src/data_process` directory contains scripts and notebooks for processing and analyzing datasets used in our research projects. This includes data cleaning, transformation, and feature extraction tasks.

## Workflow
1. Download raw phish email datasets from specified sources.
2. Follow the instructions in the notebook `phish_extract.ipynb` to clean and preprocess the data. Processed data will be saved in the `data_phish/raw` directory.
3. Use the `llm` subdirectory for scripts related to large language model (LLM) processing, such as brief captioning. Final data will be saved in the `data_phish/jsonl` directory with name `<dataset-name>_with_captions.json`.

## Data format
The final processed data (before tokenization) in `data_phish/jsonl` is stored in JSON Lines format, with the following fields:
- `text`: The full email text.
- `label`: The label indicating whether the email is benign (0) or phishing (1).
- `caption`: A brief caption summarizing the email content, generated using LLMs. (Qwen3 by default)
