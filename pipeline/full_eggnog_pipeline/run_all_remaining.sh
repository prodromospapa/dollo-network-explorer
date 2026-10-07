#!/usr/bin/env bash
set -euo pipefail

echo "Running HMM Filter..."
python3 pipeline/full_eggnog_pipeline/05_hmmsearch_filter_parallel.py

echo "HMM Filter finished! Immediately launching final steps..."
bash pipeline/full_eggnog_pipeline/run_final_steps.sh

echo "ALL DONE!"
