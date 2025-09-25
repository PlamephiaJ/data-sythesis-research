#!/usr/bin/env bash
set -euo pipefail

python3 -m dsr.cli.train "$@"
