#!/usr/bin/env bash
# ==============================================================================
# Script 1: Prepare Input Files for COUNT
#
# Converts the project's input files into the exact format required by COUNT:
#   1. Tree: NEXUS -> Newick, with quoted species names.
#   2. Matrix: CSV (Species x Genes) -> TSV (Genes x Species).
#   3. Verifies the tree and matrix have the same species.
#
# All the actual logic lives in helpers/prepare_inputs.py -- this script
# just points it at the right files.
#
# Inputs:  TCS_clean.tree, orthogroups_corHMM.csv
# Outputs: cache/tree.newick, cache/table.tsv
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$ROOT_DIR")"
CACHE_DIR="$ROOT_DIR/cache"

mkdir -p "$CACHE_DIR"

python3 "$SCRIPT_DIR/helpers/prepare_inputs.py" \
  "$PROJECT_ROOT/TCS_clean.tree" \
  "$PROJECT_ROOT/orthogroups_corHMM.csv" \
  "$CACHE_DIR/tree.newick" \
  "$CACHE_DIR/table.tsv"
