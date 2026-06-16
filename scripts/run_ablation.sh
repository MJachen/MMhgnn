#!/usr/bin/env bash
set -euo pipefail
python scripts/run_ablation.py --config_root configs/ablation "$@"
