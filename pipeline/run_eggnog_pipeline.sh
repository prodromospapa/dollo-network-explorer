#!/usr/bin/env bash
set -euo pipefail

echo "1. Preparing inputs for COUNT..."
# python3 pipeline/helpers/prepare_eggnog_inputs.py

echo "2. Running COUNT parsimony reconstruction (this may take a while)..."
./pipeline/N2_reconstruct_eggnog.sh

echo "3. Building Jaccard co-loss matrix..."
python3 pipeline/helpers/build_eggnog_jaccard.py \
    cache_eggnog/events.tsv \
    data/eggnog_dataset/network_partners.bin \
    cache_eggnog/jaccard_matrix.npy \
    cache_eggnog/jaccard_genes.tsv \
    cache_eggnog/loss_counts.npy

echo "4. Running Leiden clustering..."
python3 pipeline/N5_leiden_cluster.py \
    --matrix cache_eggnog/jaccard_matrix.npy \
    --index cache_eggnog/jaccard_genes.tsv \
    --loss-counts cache_eggnog/loss_counts.npy \
    --out-clusters data/eggnog_dataset/leiden_clusters.tsv \
    --out-summary data/eggnog_dataset/leiden_summary.tsv

echo "5. Exporting tree layout..."
python3 pipeline/helpers/export_eggnog_tree_layout.py

echo "6. Building HTML explorer..."
python3 pipeline/N6_build_html_explorer.py --dataset eggnog --output index_eggnog.html

echo "Done running EggNOG pipeline!"
