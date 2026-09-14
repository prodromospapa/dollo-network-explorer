#!/usr/bin/env python3
"""Compute the all-vs-all Jaccard similarity matrix between genes, based on
each gene's set of independent loss branches in the Dollo parsimony
reconstruction (same convention as helpers/concordance.py: Jaccard on LOSS
branches only, since Dollo parsimony assigns each gene exactly one gain).

Reads HISTORY rows directly from the Count Parsimony -history output
(test2.txt) and the family_index.tsv id->gene map, builds a boolean
(n_genes x n_nodes) loss matrix, and computes Jaccard similarity for every
gene pair via matrix multiplication:

    intersection = L @ L.T                (n_genes x n_genes)
    union        = row_sums[:,None] + row_sums[None,:] - intersection
    jaccard      = intersection / union    (0 where union == 0)

Output (compact binary, per user's choice):
    jaccard_matrix.npy   -- float32 (n_genes x n_genes) symmetric matrix
    jaccard_genes.tsv    -- row/column index -> gene name, in matrix order

Usage:
    python3 jaccard_matrix.py \
        --history ../test2.txt --families ../cache/family_index.tsv \
        --out-matrix ../results/jaccard_matrix.npy \
        --out-index ../results/jaccard_genes.tsv
"""
import argparse
import sys

import numpy as np


def load_family_index(path):
    """family_id (str, in file order) -> gene name."""
    family_ids = []
    id_to_name = {}
    with open(path) as f:
        for line in f:
            idx, name = line.rstrip("\n").split("\t")
            family_ids.append(idx)
            id_to_name[idx] = name
    return family_ids, id_to_name


def build_loss_matrix(history_path, family_ids):
    """Return (loss_matrix, n_nodes): boolean (n_genes x n_nodes) array,
    row order == family_ids order."""
    fam_to_row = {fid: i for i, fid in enumerate(family_ids)}
    n_genes = len(family_ids)

    max_nodeidx = -1
    events = []  # (row, nodeidx) pairs where loss == 1
    with open(history_path) as f:
        for line in f:
            if not line.startswith("HISTORY\t"):
                continue
            fields = line.rstrip("\n").split("\t")
            fam_id = fields[1]
            row = fam_to_row.get(fam_id)
            if row is None:
                continue
            nodeidx = int(fields[2])
            if nodeidx > max_nodeidx:
                max_nodeidx = nodeidx
            loss = fields[7]
            if loss != "0":
                events.append((row, nodeidx))

    n_nodes = max_nodeidx + 1
    loss_matrix = np.zeros((n_genes, n_nodes), dtype=bool)
    for row, nodeidx in events:
        loss_matrix[row, nodeidx] = True
    return loss_matrix, n_nodes


def compute_jaccard(loss_matrix):
    L = loss_matrix.astype(np.float32)
    intersection = L @ L.T
    row_sums = L.sum(axis=1)
    union = row_sums[:, None] + row_sums[None, :] - intersection
    jaccard = np.divide(
        intersection, union,
        out=np.zeros_like(intersection, dtype=np.float32),
        where=union > 0,
    )
    return jaccard.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--history", default="../test2.txt")
    ap.add_argument("--families", default="../cache/family_index.tsv")
    ap.add_argument("--out-matrix", default="../results/jaccard_matrix.npy")
    ap.add_argument("--out-index", default="../results/jaccard_genes.tsv")
    args = ap.parse_args()

    print("Loading family index...", file=sys.stderr)
    family_ids, id_to_name = load_family_index(args.families)
    n_genes = len(family_ids)
    print(f"  {n_genes} genes", file=sys.stderr)

    print("Building loss-branch matrix from HISTORY rows...", file=sys.stderr)
    loss_matrix, n_nodes = build_loss_matrix(args.history, family_ids)
    print(f"  {n_genes} genes x {n_nodes} nodes, "
          f"{loss_matrix.sum()} total loss events", file=sys.stderr)

    print("Computing pairwise Jaccard similarity (genes x genes)...", file=sys.stderr)
    jaccard = compute_jaccard(loss_matrix)

    print(f"Writing matrix to {args.out_matrix} ...", file=sys.stderr)
    np.save(args.out_matrix, jaccard)

    print(f"Writing gene index to {args.out_index} ...", file=sys.stderr)
    with open(args.out_index, "w") as f:
        f.write("matrix_idx\tfamily_id\tgene_name\n")
        for i, fid in enumerate(family_ids):
            f.write(f"{i}\t{fid}\t{id_to_name[fid]}\n")

    print("Done.", file=sys.stderr)
    print(f"Matrix shape: {jaccard.shape}, dtype: {jaccard.dtype}", file=sys.stderr)


if __name__ == "__main__":
    main()
