<h1 align="center">Email Data Generation using Deep Learning</h1>

<div align="center">

Project for deep-learning-based email generator.

[Full Documentation](https://plamephiaj.github.io/data-sythesis-research/design/readme.html)

</div>

# TriGuardFL: TRIPLE-LAYER DETECTION FOR BYZANTINE-ROBUST FEDERATED LEARNING AGAINST MODEL POISONING ATTACKS

[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract

**Data Sythesis Research" ....
## Tech Stack

* **Python 3.9+**
* **uv**: For fast and reliable project environment and dependency management.
* **PyTorch**
* **pyproject.toml**: For declaring project metadata and dependencies.

## Installation and Usage

This project uses `uv` for environment management to ensure a fast and consistent setup process.

### 1. Prerequisites

Ensure you have Python 3.9+ and `uv` installed on your system.

### 2. Setup Steps

```bash
# 1. Clone this repository
git clone repo_addr
cd data-sythesis-research

# 2. Create a virtual environment using uv
# This will create a .venv folder in the current directory
uv venv

# 3. Activate the virtual environment on Ubuntu 24.04
source .venv/bin/activate

# 4. Install project dependencies
# uv will automatically read the pyproject.toml file and install all dependencies
uv pip install -e .
```

If you have defined development dependencies (e.g., `pytest`) in your `pyproject.toml`, you can install them with:
```bash
uv pip install ".[dev]"
```


### 3. Running an Experiment

This project is configured with a command-line entry point called 'ds-run', which makes running experiments simple and reproducible. You no longer need to call python src/byzantine_robust_fl/main.py directly.

All parameters are controlled via command-line arguments.

Example 1: Basic MNIST experiment without attacks
This command runs a simple experiment for 10 epochs on the MNIST dataset with an MLP model.

```bash
ds-run --config configs/demo1.yaml
```
You can see all available options by running:

```bash
ds-run --help
```

## Expected Results

(Optional) You can showcase example plots or key results from your project runs here, such as:
* A comparison of model accuracy under different aggregation rules.
* The loss curve convergence with and without Byzantine attacks.

![Model Accuracy Comparison](placeholder_accuracy_plot.png)

## Contributing

Contributions of any kind are welcome! If you have a great idea or find a bug, feel free to submit a Pull Request or create an Issue.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/Feature`)
3.  Commit your Changes (`git commit -m 'Add some Feature'`)
4.  Push to the Branch (`git push origin feature/Feature`)
5.  Open a Pull Request

## 📄 License

This project is distributed under the MIT License. See the `LICENSE` file for more information.
