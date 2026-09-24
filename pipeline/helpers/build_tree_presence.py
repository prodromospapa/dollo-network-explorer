#!/usr/bin/env python3
"""Build tree_presence.<dataset>.bin for a second dataset, reusing the
existing tree_layout.json's exact gene_names/leaf_species order (tree
geometry and gene order are shared across datasets -- only presence values
differ; see scripts/build_clean_uniprot_tree.py, which builds the
orthogroup-dataset original and whose "DLTP" binary format/bit-packing this
mirrors exactly).

Usage:
    python3 build_tree_presence.py \
        --tree-layout tree_layout.json \
        --presence-pkl orthologs.pkl \
        --out tree_presence.ortholog.bin
"""
import argparse
import json
import pickle
import struct

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree-layout", required=True)
    ap.add_argument("--presence-pkl", required=True,
                     help="Gene rows x species columns presence DataFrame pickle "
                          "(e.g. orthologs.pkl), same shape as orthogroups.pkl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.tree_layout) as f:
        layout = json.load(f)
    gene_names = layout["gene_names"]
    leaf_species = layout["leaf_species"]
    n_leaves = len(leaf_species)

    with open(args.presence_pkl, "rb") as f:
        df = pickle.load(f)

    missing_genes = [g for g in gene_names if g not in df.index]
    if missing_genes:
        raise SystemExit(f"ERROR: {len(missing_genes)} tree_layout genes missing from "
                          f"{args.presence_pkl}: {missing_genes[:10]}...")
    missing_species = [s for s in leaf_species if s not in df.columns]
    if missing_species:
        raise SystemExit(f"ERROR: {len(missing_species)} tree_layout species missing from "
                          f"{args.presence_pkl}: {missing_species[:10]}...")

    df_ordered = df.loc[gene_names, leaf_species].astype(np.uint8)

    packed_bytes = bytearray()
    for gname in gene_names:
        row_bits = df_ordered.loc[gname].values
        b = bytearray((n_leaves + 7) // 8)
        for i, val in enumerate(row_bits):
            if val:
                b[i // 8] |= 1 << (i % 8)
        packed_bytes.extend(b)

    header = struct.pack("<4sIH", b"DLTP", len(gene_names), n_leaves)
    with open(args.out, "wb") as f:
        f.write(header)
        f.write(packed_bytes)
    print(f"Wrote {args.out} ({len(header) + len(packed_bytes)} bytes, "
          f"{len(gene_names)} genes x {n_leaves} leaves)")


if __name__ == "__main__":
    main()
