#!/usr/bin/env bash
set -euo pipefail
cd /home/prodromosp/dollo-network-explorer

LOGFILE="pipeline/full_eggnog_pipeline/pipeline.log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "======================================"
echo "Starting Full EggNOG Pipeline (12.5k species)"
echo "Date: $(date)"
echo "======================================"

echo "[1/8] Extracting EggNOG Ortholog Groups..."
echo "Skipping step 1 (gene_to_target_proteins.json already extracted)"

echo "[2/8] Extracting raw FASTA sequences to SQLite database..."
echo "Skipping step 2 (SQLite DB already extracted)"

echo "[3/8] Running PANTHER HMM search filter (12 threads)..."
python3 pipeline/full_eggnog_pipeline/05_hmmsearch_filter_parallel.py

echo "[4/8] Building full 12,500 species matrix..."
python3 pipeline/full_eggnog_pipeline/06_build_filtered_matrix.py

echo "[5/8] Running Dollo Parsimony inference (This will take 15-24 hours)..."
export OMP_NUM_THREADS=12
java -Xmx32g -cp tools/CountXXV.jar count.model.Parsimony \
    -gain 1000 -loss 1 \
    -history true -ancestral false \
    cache_eggnog_full/tree.newick cache_eggnog_full/table.tsv > cache_eggnog_full/reconstruction_raw.tsv 2> cache_eggnog_full/reconstruct_stderr.log

python3 pipeline/helpers/reconstruct.py cache_eggnog_full/table.tsv cache_eggnog_full/reconstruction_raw.tsv cache_eggnog_full/events.tsv

echo "[6/8] Building Jaccard matrix..."
mkdir -p data/eggnog_full_dataset
python3 pipeline/helpers/build_eggnog_jaccard.py \
    cache_eggnog_full/events.tsv \
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
cat << 'PYEOF' > pipeline/helpers/export_full_tree_layout.py
import json
from ete3 import Tree
from pathlib import Path

CACHE_DIR = Path("cache_eggnog_full")
DATA_DIR = Path("data/eggnog_full_dataset")

tree = Tree(str(CACHE_DIR / "tree.newick"), format=1)
all_nodes = [n for n in tree.traverse() if n is not tree]
tree.add_feature("_x", 0.0)
for node in tree.traverse():
    if node.up is not None:
        node.add_feature("_x", node.up._x + node.dist)

leaves_in_order = tree.get_leaves()
for i, leaf in enumerate(leaves_in_order):
    leaf.add_feature("_y", i)

for node in tree.traverse("postorder"):
    if not node.is_leaf():
        ys = [c._y for c in node.children]
        node.add_feature("_y", sum(ys) / len(ys))

nodes_out = []
for j, node in enumerate(all_nodes):
    child_ys = [c._y for c in node.children] if not node.is_leaf() else None
    nodes_out.append({
        "idx": j,
        "x": round(node._x, 5),
        "y": round(node._y, 3),
        "parent_x": round(node.up._x, 5),
        "is_leaf": node.is_leaf(),
        "species": str(node.name) if node.is_leaf() else None,
        "clade": "EggNOG Full",
        "n_panel_losses": 0,
        "children_y_min": round(min(child_ys), 3) if child_ys else None,
        "children_y_max": round(max(child_ys), 3) if child_ys else None,
    })

root_child_ys = [c._y for c in tree.children]
max_x = max(n["x"] for n in nodes_out)
root_out = {"x": 0.0, "y": tree._y, "children_y_min": round(min(root_child_ys), 3), "children_y_max": round(max(root_child_ys), 3)}

with open(DATA_DIR / "tree_layout.json", "w") as fh:
    json.dump({"nodes": nodes_out, "root": root_out, "n_leaves": len(leaves_in_order), "max_x": max_x}, fh)
PYEOF
python3 pipeline/helpers/export_full_tree_layout.py

python3 pipeline/helpers/build_eggnog_tree_presence.py cache_eggnog_full/table.tsv data/eggnog_full_dataset/tree_layout.json cache_eggnog_full/jaccard_genes.tsv data/eggnog_full_dataset/tree_presence.bin

echo "Pipeline finished at $(date)"
