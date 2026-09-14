#!/usr/bin/env bash
# ==============================================================================
# Script 3: Compute Co-Evolution Concordance (Jaccard Overlap)
#
# Ranks every gene by its evolutionary co-loss with a seed gene. Two genes
# functionally co-evolve if they are independently lost on the same
# branches of the phylogenetic tree; that overlap is scored with the
# Jaccard similarity of each gene's loss-branch set:
#
#     concordance = |seed losses AND candidate losses| / |seed losses OR candidate losses|
#
# All the logic lives in helpers/concordance.py -- see its docstring for
# the full method. This is a pure Dollo-parsimony score: no significance
# test or p-value here, just how much two genes' independent-loss
# histories overlap.
#
# Also flags whether each candidate is already a known ciliary gene per
# SYSCILIA and CiliaCarta -- a face-validity check, not a statistic (see
# helpers/concordance.py's docstring).
#
# Usage:
#   bash 03_concordance.sh SEED_GENE
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$ROOT_DIR")"
CACHE_DIR="$ROOT_DIR/cache"
RESULTS_DIR="$ROOT_DIR/results"
EVENTS="$CACHE_DIR/events.tsv"
SYSCILIA_CSV="$PROJECT_ROOT/cilia_gene_panel_scgsv1.csv"
CILIACARTA_CSV="$PROJECT_ROOT/coevolution_framework/CiliaCarta.csv"
CILIACARTA_CSV="$PROJECT_ROOT/CiliaCarta.csv"
if [[ ! -f "$CILIACARTA_CSV" && -f "$PROJECT_ROOT/junk/coevolution_framework/CiliaCarta.csv" ]]; then
  CILIACARTA_CSV="$PROJECT_ROOT/junk/coevolution_framework/CiliaCarta.csv"
fi

SEED_GENE="${1:?Usage: $0 SEED_GENE}"

if [[ ! -f "$EVENTS" ]]; then
  echo "Missing $EVENTS. Please run 02_reconstruct.sh first." >&2
  exit 1
fi

mkdir -p "$RESULTS_DIR"
OUT="$RESULTS_DIR/${SEED_GENE}_concordance.tsv"

echo "Scoring every gene's co-evolution concordance with $SEED_GENE..."
python3 "$SCRIPT_DIR/helpers/concordance.py" "$EVENTS" "$SEED_GENE" "$SYSCILIA_CSV" "$CILIACARTA_CSV" > "$OUT"

N_GENES=$(($(wc -l < "$OUT") - 1))
echo "  -> Saved: $OUT ($N_GENES genes ranked)"
echo
echo "Top 10 candidates:"
# NOT `tail -n +2 file | head -10`: head closing the pipe early after 10
# lines can send tail SIGPIPE, which is fatal under `set -o pipefail`
# (confirmed directly on this exact construct). awk reads the file
# directly and stops itself, so nothing ever writes into a pipe that
# closes early.
awk -F'\t' 'NR==1{print; next} {print; if (NR-1>=10) exit}' "$OUT" | column -t -s $'\t'
