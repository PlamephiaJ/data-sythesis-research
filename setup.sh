#!/usr/bin/env bash
# ==============================================================================
# AI Firewall - Development Environment Setup Script (Final Simplified Version)
#
# This script manages two primary lockfiles:
#     - requirements.txt: For production deployments.
#     - requirements-dev.txt: For all local development and CI/CD purposes.
#
# USAGE:
#     ./setup.sh         - Sets up the environment using existing lockfiles.
#     ./setup.sh --compile - Re-compiles lockfiles and then sets up the environment.
# ==============================================================================

set -e

# --- Configuration ---
VENV_DIR=".venv"
PROD_LOCKFILE="requirements.txt"
DEV_LOCKFILE="requirements-dev.txt"

# --- 1. Check for uv ---
echo "STEP 1: Checking for uv..."
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' command not found. Please install at https://astral.sh/uv/install.sh"
    exit 1
fi
echo "✅ uv found: $(uv --version)"

# --- 2. Create/Activate Virtual Environment ---
echo -e "\nSTEP 2: Setting up virtual environment in '${VENV_DIR}'..."
if [ ! -d "$VENV_DIR" ]; then
    uv venv
    echo "✅ Virtual environment created."
fi
source "${VENV_DIR}/bin/activate"
echo "✅ Virtual environment activated."

# --- 3. Compile Lockfiles (if requested) ---
if [[ "$1" == "--compile" ]]; then
    echo -e "\nSTEP 3: Re-compiling lockfiles..."

    echo "-> Compiling PRODUCTION lockfile (${PROD_LOCKFILE})..."
    uv pip compile pyproject.toml -o "${PROD_LOCKFILE}"

    echo "-> Compiling DEVELOPMENT lockfile (${DEV_LOCKFILE})... (Includes --all-extras)"
    uv pip compile pyproject.toml --all-extras -o "${DEV_LOCKFILE}"

    echo "✅ Both lockfiles have been successfully updated."
else
    echo -e "\nSTEP 3: Skipping compilation. Using existing lockfiles."
    if [ ! -f "${DEV_LOCKFILE}" ]; then
        echo "Error: Lockfile '${DEV_LOCKFILE}' not found." >&2
        echo "Hint: If this is the first setup, run './setup.sh --compile' to generate it." >&2
        exit 1
    fi
fi

# --- 4. Sync Development Environment (without flash-attn) ---
echo -e "\nSTEP 4: Syncing environment with '${DEV_LOCKFILE}'..."
uv pip sync "${DEV_LOCKFILE}"
echo "✅ All core dependencies are installed."

# --- 5. Manually Install Flash Attention 2 ---
echo -e "\nSTEP 5: Installing Flash Attention 2..."
uv pip install flash-attn==2.8.3 --no-build-isolation
echo "✅ Flash Attention 2 installed."

# --- 6. Install pre-commit hooks & local project ---
echo "-> Installing project in editable mode..."
uv pip install -e .
echo "✅ Local development environment is ready."
pre-commit install
echo "✅ pre-commit hooks installed successfully."

# --- 7. Final Instructions ---
echo -e "\n----------------- SETUP COMPLETE -----------------"
echo "✅ Your local environment is fully configured and ready."
echo
if [[ "$1" == "--compile" ]]; then
    echo "IMPORTANT: You have re-compiled the lockfiles. Please commit the changes:"
    echo "git add pyproject.toml ${PROD_LOCKFILE} ${DEV_LOCKFILE}"
fi
echo
echo "NEXT STEP: The environment is already activated for this session."
echo "For future sessions, run:"
echo "source .venv/bin/activate"
echo "----------------------------------------------------"
