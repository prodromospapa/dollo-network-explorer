#!/usr/bin/env bash
set -euo pipefail
cd /home/prodromosp/dollo-network-explorer

LOGFILE="pipeline/full_eggnog_pipeline/pipeline_final.log"
exec > >(tee -a "$LOGFILE") 2>&1
echo "=== Resuming Full EggNOG Pipeline at step 6 ($(date)) ==="
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
