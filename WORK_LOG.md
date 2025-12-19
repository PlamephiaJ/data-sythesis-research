<h1 align="center">Work Log for Email Data Generation using Deep Learning</h1>

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

## Dec. 19th, 2025 Update
### Yuhao:
1. Added caption and mask support in sampling code.
1. Now the model can generate emails conditioned on style captions. The first 500k iterations of training's results seem promising.
1. Solved a batch size mismatch bug in the sampling phase, the small model has the length 1024 while the evaluation dataloader config has length 2048. Solved by clamping the length to the model's max length.
1. Plan for the coming Christmas holidays:
    - Run the training and observe the sampling results.
    - Neaten the codebase to be more maintainable.

## Dec. 18th, 2025 Update
### Yuhao:
1. Bug fixes in flash attention implementation with masked input.
1. Add classifer-free guidance support for training.
1. The code is now able to run training with masked input and classifier-free guidance. First experiments will be conducted.
1. Add `tensorboard` logging. Dependency added in `pyproject.toml`.

## Dec. 16th, 2025 Update
### Yuhao:
1. Tested the data loading pipeline for the phishing email dataset.
    - The processed data contains `text`, `text_mask`, `style_caption`, `style_caption_mask` fields.
1. Modify code to support masked input during training, including updates to graph, transformer, and score entropy calculation, etc.
1. Currently, the condition is modulated with time step embedding. The effect is going to be investigated.
1. All closure design patterns are refactored to class-based design patterns for better readability and maintainability.

## Dec. 15th, 2025 Update
### Yuhao:
- Finished data processing and loading for the phishing email dataset.
- Now the phishing email dataset can be used for unconditional training.
- Updated `configs/data/default.yaml` to include the phishing email dataset as the default dataset.
- Updated `src/train/run_train.py` to accommodate the new data format with `text_input_ids`.

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
