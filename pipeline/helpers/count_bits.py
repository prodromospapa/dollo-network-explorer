#!/usr/bin/env python3
"""Convert COUNT's text history output into packed bit matrices.

COUNT (-history true) writes one text line per (family, tree node) pair:

    HISTORY  family_idx  nodeidx  presence  multi  maxpresence  gain  loss  ...

For the full EggNOG run that is ~278M lines (~12 GB). Under Dollo parsimony
on a 0/1 table, presence, gain and loss are each 0/1, so they are stored
here as np.packbits matrices of shape (n_families, ceil(n_nodes / 8)):
row = family (table.tsv row order), bit = nodeidx. ~35 MB per matrix.

Usage (convert an already-merged raw file):
    python3 count_bits.py TABLE_TSV RAW_COUNT_OUTPUT OUT_NPZ
"""
import os
import sys

import numpy as np
import pandas as pd

FIELDS = ("presence", "gain", "loss")
# Column positions in a HISTORY line (see reconstruct.py).
COLS = {"family": 1, "nodeidx": 2, "presence": 3, "gain": 6, "loss": 7}


def history_to_bits(raw_path, n_families):
    """Parse COUNT history text -> (n_nodes, {field: packed bits}).

    Fails loudly on anything unexpected (non-0/1 values, missing or
    duplicate rows) instead of producing a silently wrong matrix."""
    hits = {f: [] for f in FIELDS}  # (family, node) pairs where field == 1
    n_rows, max_node = 0, -1
    # '#' lines (header '#|' block, '#SCORE') are skipped; the first data line
    # is COUNT's 'HISTORY family nodeidx ...' column header.
    reader = pd.read_csv(raw_path, sep="\t", comment="#", header=0,
                         usecols=list(COLS.values()), dtype=np.int32,
                         chunksize=20_000_000)
    for df in reader:
        a = {k: df.iloc[:, sorted(COLS.values()).index(c)].to_numpy()
             for k, c in COLS.items()}
        if a["family"].min() < 0 or a["family"].max() >= n_families:
            sys.exit(f"{raw_path}: family index outside 0..{n_families - 1}")
        for f in FIELDS:
            v = a[f]
            if v.min() < 0 or v.max() > 1:
                sys.exit(f"{raw_path}: {f} has values outside 0/1")
            m = v == 1
            hits[f].append((a["family"][m], a["nodeidx"][m]))
        n_rows += len(df)
        max_node = max(max_node, int(a["nodeidx"].max()))

    n_nodes = max_node + 1
    if n_rows != n_families * n_nodes:
        sys.exit(f"{raw_path}: {n_rows} rows, expected "
                 f"{n_families} families x {n_nodes} nodes")

    bits = {}
    for f in FIELDS:
        dense = np.zeros((n_families, n_nodes), dtype=bool)
        for fam, node in hits[f]:
            dense[fam, node] = True
        bits[f] = np.packbits(dense, axis=1)
    return n_nodes, bits


def save_npz(path, genes, n_nodes, bits):
    """Write atomically, so a killed run never leaves a valid-looking file."""
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, genes=np.asarray(genes, dtype=str),
                        n_nodes=n_nodes, **bits)
    os.replace(tmp, path)


def load_npz(path):
    """-> (genes, n_nodes, {field: packed bits})."""
    z = np.load(path)
    return list(z["genes"]), int(z["n_nodes"]), {f: z[f] for f in FIELDS}


def unpack(packed, n_nodes):
    """Packed bits -> (n_families, n_nodes) bool matrix."""
    return np.unpackbits(packed, axis=1, count=n_nodes).astype(bool)


def table_genes(table_path):
    """Gene names in table.tsv row order (= COUNT's family_idx)."""
    with open(table_path) as f:
        next(f)  # header
        return [line.split("\t", 1)[0] for line in f]


def main():
    table_path, raw_path, out_path = sys.argv[1:4]
    genes = table_genes(table_path)
    n_nodes, bits = history_to_bits(raw_path, len(genes))
    save_npz(out_path, genes, n_nodes, bits)
    print(f"Saved {out_path}: {len(genes)} families x {n_nodes} nodes; "
          + ", ".join(f"{f}={int(np.unpackbits(bits[f]).sum())}" for f in FIELDS))


if __name__ == "__main__":
    main()
