#!/usr/bin/env bash
# Regression tests for the bash-only Dollo pipeline, run against the toy
# fixtures (tests/fixtures/toy.newick + toy_table.txt / toy_table2.txt).
#
# Covers two real bugs found and fixed during development, so they don't
# silently regress:
#
#   1. `tail -n +2 file | head -n N` under `set -o pipefail` dies with
#      SIGPIPE (exit 141) when N is fewer than the file's total lines --
#      head closing the pipe early kills tail's write, and pipefail
#      propagates that as a fatal script error. Confirmed on the real
#      ~19,758-row concordance file. Fixed by using awk to both read and
#      truncate in one process instead of a two-process pipe.
#
#   2. In `04_significance.sh`'s per-permutation-batch parser, COUNT's raw
#      stdout format is `HISTORY  family_idx  nodeidx  ...` -- column 1 is
#      literally the word "HISTORY" on every row, not the family/
#      permutation identifier. An early version of the script used $1 as
#      the permutation boundary marker (copied from events.tsv's already-
#      relabeled format), which never changes and so never triggers a
#      boundary -- collapsing ~1000 permutations into ~14 emitted scores.
#
#   3. The original awk `03_concordance.sh` streams through events.tsv in a
#      SINGLE pass, building the seed gene's loss-branch set as it goes.
#      Any candidate gene whose rows appear BEFORE the seed gene's rows in
#      the file gets scored against an still-empty seed-loss set, silently
#      reporting shared_losses=0 (and therefore concordance=0) even when
#      the true overlap is nonzero. Confirmed on the real dataset: 8,666 of
#      19,757 genes had wrong scores for seed gene SCAPER this way,
#      including at least two real top-30 hits (STK36, IFT172) that were
#      silently dropped from the ranking entirely. The Python rewrite in
#      scripts_simple/helpers/concordance.py loads the whole file before
#      scoring anything, which does not have this ordering dependency.
#
# Usage: bash tests/run_tests.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
FIXTURES="$SCRIPT_DIR/fixtures"

PASS=0
FAIL=0

check() {
  local desc="$1" got="$2" want="$3"
  if [[ "$got" == "$want" ]]; then
    echo "  ok   - $desc"
    PASS=$((PASS+1))
  else
    echo "  FAIL - $desc (want [$want], got [$got])"
    FAIL=$((FAIL+1))
  fi
}

echo "=== Test group 1: DolloDriver-equivalent reconstruction (COUNT CLI, high gain penalty) on toy tree ==="
TMPDIR1="$(mktemp -d)"
trap 'rm -rf "$TMPDIR1"' EXIT

java -cp "$ROOT_DIR/tools/CountXXV.jar" count.model.Parsimony \
  -gain 1000 -loss 1 -history true -ancestral false \
  "$FIXTURES/toy.newick" "$FIXTURES/toy_table.txt" \
  > "$TMPDIR1/raw.tsv" 2>/dev/null

# toy_table.txt: family order is gene_universal, gene_lostinCD, gene_lostinD,
# gene_lostinAB_and_D, gene_absent (0-indexed families 0-4).
gene_universal_events=$(awk -F'\t' '$1=="HISTORY" && $2==0 {g+=$7; l+=$8} END{print g+l}' "$TMPDIR1/raw.tsv")
check "gene_universal: 1 event (gain only, no losses)" "$gene_universal_events" "1"

gene_lostinCD_events=$(awk -F'\t' '$1=="HISTORY" && $2==1 {g+=$7; l+=$8} END{print g+l}' "$TMPDIR1/raw.tsv")
check "gene_lostinCD: 1 event (single late gain preferred over gain+2 losses)" "$gene_lostinCD_events" "1"

gene_lostinD_events=$(awk -F'\t' '$1=="HISTORY" && $2==2 {g+=$7; l+=$8} END{print g+l}' "$TMPDIR1/raw.tsv")
check "gene_lostinD: 2 events (gain + 1 loss on D only, not CD ancestor)" "$gene_lostinD_events" "2"

gene_lostinAB_and_D_events=$(awk -F'\t' '$1=="HISTORY" && $2==3 {g+=$7; l+=$8} END{print g+l}' "$TMPDIR1/raw.tsv")
check "gene_lostinAB_and_D: 1 event (late gain on C beats gain+2 losses)" "$gene_lostinAB_and_D_events" "1"

gene_absent_events=$(awk -F'\t' '$1=="HISTORY" && $2==4 {g+=$7; l+=$8} END{print g+l}' "$TMPDIR1/raw.tsv")
check "gene_absent: 0 events" "$gene_absent_events" "0"

echo
echo "=== Test group 2: 03_concordance.sh on toy_table2.txt (regression: uninitialized n_cur_loss printed blank; head/tail pipe dropped rows) ==="
TMPDIR2="$(mktemp -d)"
mkdir -p "$TMPDIR2/cache" "$TMPDIR2/tools"
cp "$ROOT_DIR/tools/CountXXV.jar" "$TMPDIR2/tools/"
cp "$FIXTURES/toy.newick" "$TMPDIR2/cache/tree.newick"
cp "$FIXTURES/toy_table2.txt" "$TMPDIR2/cache/table.tsv"

java -cp "$TMPDIR2/tools/CountXXV.jar" count.model.Parsimony \
  -gain 1000 -loss 1 -history true -ancestral false \
  "$TMPDIR2/cache/tree.newick" "$TMPDIR2/cache/table.tsv" \
  > "$TMPDIR2/cache/reconstruction_raw.tsv" 2>/dev/null

awk -F'\t' 'NR>1{print $1}' "$TMPDIR2/cache/table.tsv" > "$TMPDIR2/.family_names.txt"
awk -F'\t' -v names_file="$TMPDIR2/.family_names.txt" '
  BEGIN {
    i = 0
    while ((getline name < names_file) > 0) { family_name[i] = name; i++ }
    print "gene\tnodeidx\tpresence\tgain\tloss"
  }
  $1 == "HISTORY" && $2 != "family" {
    idx = $2 + 0
    print family_name[idx] "\t" $3 "\t" $4 "\t" $7 "\t" $8
  }
' "$TMPDIR2/cache/reconstruction_raw.tsv" > "$TMPDIR2/cache/events.tsv"

mkdir -p "$TMPDIR2/scripts"
cp "$SCRIPT_DIR/../scripts/03_concordance.sh" "$TMPDIR2/scripts/"
CONCORDANCE_OUT="$(cd "$TMPDIR2" && bash scripts/03_concordance.sh partial_gene)"

n_rows=$(echo "$CONCORDANCE_OUT" | tail -n +2 | wc -l)
check "all 3 non-seed genes present in output (no rows dropped)" "$n_rows" "3"

seedgene_row=$(echo "$CONCORDANCE_OUT" | awk -F'\t' '$1=="seedgene"')
seedgene_n_loss=$(echo "$seedgene_row" | awk -F'\t' '{print $2}')
check "seedgene (0 real losses) prints '0', not blank" "$seedgene_n_loss" "0"

rm -rf "$TMPDIR2"

echo
echo "=== Test group 3: 04_significance.sh permutation-batch parser (regression: \$1 is always \"HISTORY\", not the permutation id) ==="
TMPDIR3="$(mktemp -d)"
mkdir -p "$TMPDIR3/cache" "$TMPDIR3/tools" "$TMPDIR3/scripts" "$TMPDIR3/results"
cp "$ROOT_DIR/tools/CountXXV.jar" "$TMPDIR3/tools/"
cp "$FIXTURES/toy.newick" "$TMPDIR3/cache/tree.newick"
cp "$FIXTURES/toy_table2.txt" "$TMPDIR3/cache/table.tsv"
cp "$ROOT_DIR/scripts/"*.sh "$TMPDIR3/scripts/"

java -cp "$TMPDIR3/tools/CountXXV.jar" count.model.Parsimony \
  -gain 1000 -loss 1 -history true -ancestral false \
  "$TMPDIR3/cache/tree.newick" "$TMPDIR3/cache/table.tsv" \
  > "$TMPDIR3/cache/reconstruction_raw.tsv" 2>/dev/null
awk -F'\t' 'NR>1{print $1}' "$TMPDIR3/cache/table.tsv" > "$TMPDIR3/.fn.txt"
awk -F'\t' -v names_file="$TMPDIR3/.fn.txt" '
  BEGIN { i=0; while ((getline name < names_file) > 0) { family_name[i]=name; i++ }; print "gene\tnodeidx\tpresence\tgain\tloss" }
  $1=="HISTORY" && $2!="family" { idx=$2+0; print family_name[idx] "\t" $3 "\t" $4 "\t" $7 "\t" $8 }
' "$TMPDIR3/cache/reconstruction_raw.tsv" > "$TMPDIR3/cache/events.tsv"

(cd "$TMPDIR3" && bash scripts/04_significance.sh partial_gene 3 100 > /tmp/test3_sig.log 2>&1)
SIG_OUT="$TMPDIR3/results/partial_gene_significance.tsv"

if [[ -f "$SIG_OUT" ]]; then
  n_result_rows=$(tail -n +2 "$SIG_OUT" | wc -l)
  check "significance test produces one row per top candidate" "$n_result_rows" "3"
else
  echo "  FAIL - significance test did not produce output file"
  FAIL=$((FAIL+1))
  cat /tmp/test3_sig.log
fi

rm -rf "$TMPDIR3"

echo
echo "=== Test group 4: scripts_simple/helpers/concordance.py file-order independence (regression: awk's single-pass streaming missed overlap for genes appearing before the seed gene) ==="
TMPDIR4="$(mktemp -d)"
mkdir -p "$TMPDIR4/cache" "$TMPDIR4/tools"
cp "$ROOT_DIR/tools/CountXXV.jar" "$TMPDIR4/tools/"
cp "$FIXTURES/toy.newick" "$TMPDIR4/cache/tree.newick"
# toy_table_order.txt: "before_seed_gene" appears BEFORE "seedgene" in file
# order, and has an IDENTICAL presence pattern (both lost in C,D) -- so
# their true Jaccard concordance is 1.0. A single-pass streaming scorer
# that hasn't loaded the seed's losses yet when it reaches an earlier gene
# would wrongly report 0 here.
cp "$FIXTURES/toy_table_order.txt" "$TMPDIR4/cache/table.tsv"

java -cp "$TMPDIR4/tools/CountXXV.jar" count.model.Parsimony \
  -gain 1000 -loss 1 -history true -ancestral false \
  "$TMPDIR4/cache/tree.newick" "$TMPDIR4/cache/table.tsv" \
  > "$TMPDIR4/cache/reconstruction_raw.tsv" 2>/dev/null
python3 "$SCRIPT_DIR/../scripts_simple/helpers/reconstruct.py" \
  "$TMPDIR4/cache/table.tsv" "$TMPDIR4/cache/reconstruction_raw.tsv" "$TMPDIR4/cache/events.tsv" > /dev/null

CONCORDANCE_OUT="$(python3 "$SCRIPT_DIR/../scripts_simple/helpers/concordance.py" "$TMPDIR4/cache/events.tsv" seedgene)"
before_seed_jaccard=$(echo "$CONCORDANCE_OUT" | awk -F'\t' '$1=="before_seed_gene"{print $5}')
check "gene appearing before the seed in file order still scores correctly (Jaccard=1)" "$before_seed_jaccard" "1"

rm -rf "$TMPDIR4"

echo
echo "=================================="
echo "Results: $PASS passed, $FAIL failed"
echo "=================================="
exit $([ "$FAIL" -eq 0 ] && echo 0 || echo 1)
