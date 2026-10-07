#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
CACHE_DIR="$ROOT_DIR/cache_eggnog"
COUNT_JAR="$ROOT_DIR/tools/CountXXV.jar"

TREE="$CACHE_DIR/tree.newick"
TABLE="$CACHE_DIR/table.tsv"
RAW_OUT="$CACHE_DIR/reconstruction_raw.tsv"
EVENTS_OUT="$CACHE_DIR/events.tsv"

if [ ! -f "$COUNT_JAR" ]; then
    echo "Error: tools/CountXXV.jar not found at $COUNT_JAR"
    exit 1
fi

echo "Running COUNT parsimony reconstruction on EggNOG dataset..."
java -Xmx8g -cp "$COUNT_JAR" count.model.Parsimony \
  -gain 1000 -loss 1 \
  -history true -ancestral false \
  "$TREE" "$TABLE" > "$RAW_OUT" 2> "$CACHE_DIR/reconstruct_stderr.log"

echo "Formatting events..."
python3 "$SCRIPT_DIR/helpers/reconstruct.py" "$TABLE" "$RAW_OUT" "$EVENTS_OUT"

echo "Done! Output saved to $CACHE_DIR/events.tsv"
