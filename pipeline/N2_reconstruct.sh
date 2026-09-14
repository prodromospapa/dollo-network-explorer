#!/usr/bin/env bash
# ==============================================================================
# Script 2: Dollo Ancestral State Reconstruction
#
# Reconstructs the evolutionary history of gene gain and loss for every gene
# across all branches of the species tree using COUNT.
#
# Dollo's Law: a gene is gained at most once (single origin) and can be lost
# independently many times. COUNT's generalized parsimony minimizes total
# event cost = gain_penalty * n_gains + loss_penalty * n_losses. Setting
# gain_penalty=1000 and loss_penalty=1 enforces this: our 196-species tree
# has 389 branches, so the most losses any gene could ever have is 389.
# Since 1000 > 389, a second gain is NEVER cheaper than paying for losses.
#
# Formatting COUNT's raw output into a clean events table is done by
# helpers/reconstruct.py -- this script just runs COUNT and calls it.
#
# Inputs:  cache/tree.newick, cache/table.tsv
# Outputs: cache/events.tsv (gene, nodeidx, presence, gain, loss)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
CACHE_DIR="$ROOT_DIR/cache"
JAR="$ROOT_DIR/tools/CountXXV.jar"

TREE="$CACHE_DIR/tree.newick"
TABLE="$CACHE_DIR/table.tsv"
RAW_OUT="$CACHE_DIR/reconstruction_raw.tsv"
EVENTS_OUT="$CACHE_DIR/events.tsv"

GAIN_PENALTY=1000
LOSS_PENALTY=1

if [[ ! -f "$TREE" || ! -f "$TABLE" ]]; then
  echo "Missing $TREE or $TABLE. Please run 01_prepare_inputs.sh first." >&2
  exit 1
fi

echo "[Step 1/2] Running COUNT Dollo reconstruction across all genes..."
echo "  Command: java -cp CountXXV.jar count.model.Parsimony -gain $GAIN_PENALTY -loss $LOSS_PENALTY ..."

java -cp "$JAR" count.model.Parsimony \
  -gain "$GAIN_PENALTY" -loss "$LOSS_PENALTY" \
  -history true -ancestral false \
  "$TREE" "$TABLE" > "$RAW_OUT" 2> "$CACHE_DIR/reconstruct_stderr.log"

echo "  -> Raw reconstruction complete ($(wc -l < "$RAW_OUT") lines)."

python3 "$SCRIPT_DIR/helpers/reconstruct.py" "$TABLE" "$RAW_OUT" "$EVENTS_OUT"
