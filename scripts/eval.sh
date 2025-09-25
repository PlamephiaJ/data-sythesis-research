#!/usr/bin/env bash
set -euo pipefail

python3 -m pytest tests/test_smoke.py "$@"
