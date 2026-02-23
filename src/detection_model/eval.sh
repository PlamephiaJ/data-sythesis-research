#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -f ".venv/bin/activate" ]]; then
	source .venv/bin/activate
fi

SWEEP_DIR="exp_local/detection_model/2026.02.23/180306"
INPUT_FILE="data_phish/eval/Nazario_cleaned_raw.json"

if [[ ! -d "$SWEEP_DIR" ]]; then
	echo "Sweep directory not found: $SWEEP_DIR" >&2
	exit 1
fi

echo "Sweep eval root: $SWEEP_DIR"
echo "Input file: $INPUT_FILE"

success=0
failed=0

for run_dir in "$SWEEP_DIR"/*; do
	if [[ ! -d "$run_dir" ]]; then
		continue
	fi

	name="$(basename "$run_dir")"
	if [[ "$name" == "tensorboard" ]]; then
		continue
	fi

	echo "========================================"
	echo "Evaluating: $run_dir"

	if python src/detection_model/eval.py --model_path "$run_dir" --input_file "$INPUT_FILE"; then
		((success+=1))
		echo "[OK] $run_dir"
	else
		((failed+=1))
		echo "[FAIL] $run_dir" >&2
	fi
done

echo "========================================"
echo "Finished. success=$success failed=$failed"

if [[ $failed -gt 0 ]]; then
	exit 1
fi
