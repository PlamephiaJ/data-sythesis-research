<h1 align="center">Email Data Generation using Deep Learning</h1>

<div align="center">

Project for deep-learning-based email generator.

[Full Documentation](https://plamephiaj.github.io/data-sythesis-research/design/readme.html)

</div>

# Data Sythesis Research: Rule-guided Discrete Diffusion for Email Generation

[![Python Version](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Soft Links in this project
- data --> /home/shared_folder/shared_datasets/data-sythesis-research/data
- data_phish --> /home/shared_folder/shared_datasets/data-sythesis-research/data_phish

# Reverse Chronological Order

## Dec. 10th, 2025 Update
### Yuhao:
- Refactored data loading code in `src/data_process/data.py` to decouple dataset loading and data processing.
- Introduced `configs/data/*.yaml` to manage data-related configurations.
- Updated `configs/config.yaml` and dataset-specific config files to align with the new data configuration structure.
- The original data loading logic is mainly used for language modeling tasks, which can only support chunk-based data formats. The newly added structure allows for condition - text entry based data formats.

## Dec. 5th, 2025 Update
### Yuhao:
- Added comments in source code to clarify details of the score entropy discrete diffusion model.
- Methods studies in progress.

## Dec. 1st, 2025 Update
### Yuhao:
- Updated `src/data_process/README.md` to include detailed workflow for data processing.
- Added note in `src/data_process/llm/style_caption.py` about Ollama's speed and future work considerations.
- Created `WORK_LOG.md` to track changes and updates in the project.
- Finished examining the extracted dataset in `data_phish/jsonl`. Now we have 203,793 emails with brief captions. `Overall benign=109465 phish=94328  phish_ratio=46.29% total_samples=203793`
