#!/usr/bin/env python3
"""
Rebuild network_partners_eggnog.bin with presence/absence Jaccard scores.

Partner score = |species with both genes| / |species with either gene|,
computed over the leaves in tree_presence_eggnog.bin. Per-gene loss counts
stay the Dollo (COUNT) values; Leiden modules are unaffected.

Run after export_eggnog_circular_layout.py (needs tree_presence_eggnog.bin).
Usage (from repo root):
  python3 pipeline/helpers/build_presence_partners.py
"""

import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline"))
from N6_build_html_explorer_filtered import generate_partners_binary  # noqa: E402

TOP_K = 500
JACCARD_FLOOR = 0.08


def main():
    names = json.loads((ROOT / "data_eggnog.json").read_text())["names"]

    raw = (ROOT / "tree_presence_eggnog.bin").read_bytes()
    assert raw[:4] == b"DLTP"
    n_genes, n_leaves = struct.unpack("<IH", raw[4:10])
    assert n_genes == len(names)
    bpg = math.ceil(n_leaves / 8)
    packed = np.frombuffer(raw, dtype=np.uint8, offset=10).reshape(n_genes, bpg)
    P = np.unpackbits(packed, axis=1, count=n_leaves, bitorder="little").astype(np.float32)

    print(f"Presence Jaccard over {n_genes} genes x {n_leaves} species...")
    inter = P @ P.T
    sizes = P.sum(axis=1)
    union = sizes[:, None] + sizes[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        J = np.where(union > 0, inter / union, 0.0).astype(np.float32)
    np.save(ROOT / "cache_eggnog_full" / "presence_jaccard.npy", J)

    # Dollo loss counts, re-ordered to names order
    loss_counts = np.load(ROOT / "cache_eggnog_full" / "loss_counts.npy")
    with open(ROOT / "cache_eggnog_full" / "jaccard_genes.tsv") as f:
        next(f)
        dollo_idx = {p[2]: int(p[0]) for p in (line.rstrip("\n").split("\t") for line in f)}
    losses = np.array([loss_counts[dollo_idx[g]] for g in names])

    name_to_id = {g: i for i, g in enumerate(names)}
    out = ROOT / "network_partners_eggnog.bin"
    generate_partners_binary(out, J, names, name_to_id, name_to_id,
                             dict(enumerate(names)), losses, TOP_K, JACCARD_FLOOR)


if __name__ == "__main__":
    main()
