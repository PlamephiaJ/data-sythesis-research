# Data Processing Workspace

The `src/data_process` directory contains scripts and notebooks for processing and analyzing datasets used in our research projects. This includes data cleaning, transformation, and feature extraction tasks.

## Workflow
1. Download raw phish email datasets from specified sources.
2. Follow the instructions in the notebook `phish_extract.ipynb` to clean and preprocess the data. Processed data will be saved in the `data_phish/raw` directory.
3. Use the `llm` subdirectory for scripts related to large language model (LLM) processing, such as brief captioning. Final data will be saved in the `data_phish/jsonl` directory with name `<dataset-name>_with_captions.json`.
