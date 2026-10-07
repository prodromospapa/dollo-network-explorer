#!/usr/bin/env bash
set -euo pipefail
cd /home/prodromosp/dollo-network-explorer

LOGFILE="pipeline/full_eggnog_pipeline/pipeline_final.log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "======================================"
echo "Resuming Full EggNOG Pipeline - Final Steps"
echo "Date: $(date)"
echo "======================================"

echo "[4/8] Building full 12,500 species matrix from chunks..."
python3 pipeline/full_eggnog_pipeline/06_build_filtered_matrix.py

echo "[5/8] Running Dollo Parsimony inference..."
# Output is packed presence/gain/loss bits (~100 MB), not COUNT's ~12 GB text.
python3 pipeline/helpers/parallel_count.py cache_eggnog_full/table.tsv cache_eggnog_full/tree.newick cache_eggnog_full/history_bits.npz

echo "[6/8] Building Jaccard matrix..."
mkdir -p data/eggnog_full_dataset
python3 pipeline/helpers/build_eggnog_jaccard.py \
    cache_eggnog_full/history_bits.npz \
    data/eggnog_full_dataset/network_partners.bin \
    cache_eggnog_full/jaccard_matrix.npy \
    cache_eggnog_full/jaccard_genes.tsv \
    cache_eggnog_full/loss_counts.npy

echo "[7/8] Leiden clustering..."
python3 pipeline/N5_leiden_cluster.py \
    --matrix cache_eggnog_full/jaccard_matrix.npy \
    --index cache_eggnog_full/jaccard_genes.tsv \
    --loss-counts cache_eggnog_full/loss_counts.npy \
    --out-dir data/eggnog_full_dataset

echo "[8/8] Building final interactive HTML..."
python3 pipeline/helpers/export_full_tree_layout.py
python3 pipeline/helpers/build_eggnog_tree_presence.py cache_eggnog_full/table.tsv data/eggnog_full_dataset/tree_layout.json cache_eggnog_full/jaccard_genes.tsv data/eggnog_full_dataset/tree_presence.bin

echo "Pipeline fully finished at $(date)"
