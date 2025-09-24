<h1 align="center">Email Data Generation using Deep Learning</h1>

<div align="center">

Project for deep-learning-based email generator.

[Full Documentation](https://plamephiaj.github.io/data-sythesis-research/design/readme.html)

</div>

# Data Sythesis Research: Rule-guided Discrete Diffusion for Email Generation

[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract

This project aims to design and implement a deep learning framework that leverages both Diffusion Models and Large Language Models (LLMs) to collaboratively generate “phishing-style” email samples, enabling security researchers to test/evaluate email filters and train anti-phishing detection models.

With the development of deep learning technology, using deep learning—especially advanced large language models—for phishing email detection has become a promising solution. Data is fundamental to driving the progress of deep learning models, but due to the sensitive nature of phishing emails, which often contain a large amount of malicious content and sensitive information, obtaining such data is much more difficult than for other types. Existing open-source phishing email datasets have undergone various anonymization processes, resulting in the loss of original information, and most of the data is outdated and can no longer meet the needs of rapidly evolving phishing strategies. This work aims to use diffusion models and LLMs to generate phishing emails, providing high-quality training data for downstream detection models.
## Tech Stack

* **Python 3.9–3.12**
* **PyTorch** + CUDA for distributed training
* **Hydra/OmegaConf** for hierarchical configuration
* **uv** (*optional*) for environment and dependency management

## Installation and Usage

This project uses `uv` for environment management to ensure a fast and consistent setup process.

### 1. Prerequisites

Ensure you have Python 3.13+ and `uv` installed on your system.

### 2. Setup Steps

```bash
# 1. Clone this repository
git clone https://github.com/PlamephiaJ/data-sythesis-research
cd data-sythesis-research

# 2. Create a virtual environment using uv
# This will create a .venv folder in the current directory
uv venv

# 3. Activate the virtual environment on Ubuntu 24.04
source .venv/bin/activate

# 4. Install project dependencies
# uv will automatically read the pyproject.toml file and install all dependencies
uv pip install -e .

# 5. Install flash attention manually
uv add flash-attn --no-build-isolation
```

If you have defined development dependencies (e.g., `pytest`) in your `pyproject.toml`, you can install them with:
```bash
uv pip install ".[dev]"
```


### 3. Data & Outputs

The repository keeps the working directories inside the workspace:

* `data/` – structured into `raw/`, `interim/`, and `processed/`. Populate `data/raw/` with the corpora you plan to train on or link it to shared datasets.
* `outputs/` – Hydra writes experiment artifacts here (checkpoints, logs, samples). Each run gets its own timestamped subdirectory.


### 4. Running an Experiment

The refactored package exposes a small set of import-friendly CLI tools. Launch them directly or reuse the helper scripts under `scripts/`.

```bash
# Launch a Hydra-driven training job
python3 -m dsr.cli.train
# or simply
./scripts/train.sh

# Generate samples (supports optional --prefix/--suffix constraints)
python3 -m dsr.cli.sample --model_path louaaron/sedd-medium
# or use the wrapper
./scripts/sample.sh --steps 512 --batch_size 4

# Run the smoke test suite (requires `uv pip install ".[dev]"`)
./scripts/eval.sh
```

Hydra config groups live under `configs/`:

* `configs/dataset/` – data sources (`dataset=openwebtext` by default)
* `configs/model/` – model architecture presets (`model=transformer_small`/`transformer_medium`)
* `configs/train/` – training, evaluation, and optimization schedules

Override any value on the command line, for example:

```bash
python3 -m dsr.cli.train model=transformer_medium train.training.batch_size=128
```

### 5. Project Layout

```
data-sythesis-research/
├── configs/            # Hydra configuration groups
├── scripts/            # Convenience wrappers for CLI entry points
├── src/dsr/            # Python package (data, models, train, utils, cli)
├── tests/              # Smoke tests
├── data/               # (Git-ignored) datasets with .gitkeep placeholders
├── outputs/            # (Git-ignored) experiment artifacts
└── notebooks/          # Exploratory analysis notebooks
```

### 6. Testing

A lightweight smoke test ensures the sampler registry stays wired up:

```bash
./scripts/eval.sh
# internally runs: python3 -m pytest tests/test_smoke.py
```

## Monitoring Runs

Each Hydra run writes logs to `outputs/<timestamp>/logs` and checkpoints to `outputs/<timestamp>/checkpoints/`. Sample generations are saved under `outputs/<timestamp>/samples/` when `training.snapshot_sampling=true`.

Use the console logs or inspect the JSON/`train.log` files under the run directory to track loss and evaluation metrics. Integrations such as Weights & Biases can be re-enabled by pointing the code to your account in `configs/train/base.yaml`.

## Contributing

Contributions of any kind are welcome! If you have a great idea or find a bug, feel free to submit a Pull Request or create an Issue.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/Feature`)
3.  Commit your Changes (`git commit -m 'Add some Feature'`)
4.  Push to the Branch (`git push origin feature/Feature`)
5.  Open a Pull Request

## 📄 License

This project is distributed under the MIT License. See the `LICENSE` file for more information.
