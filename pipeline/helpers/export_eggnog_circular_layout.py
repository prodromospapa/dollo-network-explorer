#!/usr/bin/env python3
"""
Export the full EggNOG species tree in the circular-layout format that the
explorer's tree view expects (same schema as the root tree_layout.json), plus
the matching per-gene presence bitmap.

Outputs (repo root):
  tree_layout_eggnog.json   circular layout, NCBI-taxonomy clade colouring
  tree_presence_eggnog.bin  "DLTP" | uint32 n_genes | uint16 n_leaves |
                            ceil(n_leaves/8) bytes per gene, LSB-first,
                            genes in data_eggnog.json "names" order

Nodes carry a contiguous leaf range [l0, l1] instead of an explicit leaf list
(12.5k leaves would make per-node lists tens of MB).

Usage (from repo root):
  python3 pipeline/helpers/export_eggnog_circular_layout.py
"""

import json
import math
import struct
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from ete3 import NCBITaxa, Tree

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "cache_eggnog_full"
LAYOUT_OUT = ROOT / "tree_layout_eggnog.json"
PRESENCE_OUT = ROOT / "tree_presence_eggnog.bin"
DATA_JSON = ROOT / "data_eggnog.json"

START_DEG = 90.0
GAP_DEG = 0.0
R_ROOT = 35.0
R_TREE = 300.0

# Taxonomy levels exposed in the UI selector -> NCBI ranks to try, in order.
LEVEL_RANKS = {
    "supergroup": ["superkingdom", "domain"],
    "kingdom": ["kingdom", "superkingdom", "domain"],
    "phylum": ["phylum", "kingdom", "superkingdom", "domain"],
    "detailed": ["class", "phylum", "kingdom", "superkingdom", "domain"],
    "tcs": ["phylum", "kingdom", "superkingdom", "domain"],
}

PALETTE = [
    "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f",
    "#e5c494", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#17becf", "#bcbd22", "#f781bf", "#14b8a6", "#84cc16",
    "#f59e0b", "#6366f1",
]
OTHER_COLOR = "#94a3b8"


def leaf_taxonomy(ncbi, taxids):
    lineages = {}
    all_ids = set()
    for t in taxids:
        try:
            lin = ncbi.get_lineage(int(t)) or []
        except Exception:
            lin = []
        lineages[t] = lin
        all_ids.update(lin)
    ranks = ncbi.get_rank(list(all_ids))
    names = ncbi.get_taxid_translator(list(all_ids | {int(t) for t in taxids if t.isdigit()}))

    out = {}
    for t, lin in lineages.items():
        by_rank = {ranks.get(i): names.get(i) for i in lin}
        info = {"sci_name": names.get(int(t), t) if t.isdigit() else t}
        for level, candidates in LEVEL_RANKS.items():
            info[level] = next((by_rank[r] for r in candidates if by_rank.get(r)), "Unclassified")
        out[t] = info
    return out


def assign_colors(values):
    counts = Counter(values)
    colors = {}
    for i, (v, _) in enumerate(counts.most_common()):
        colors[v] = PALETTE[i] if i < len(PALETTE) and v != "Unclassified" else OTHER_COLOR
    return colors


def clade_blocks(leaf_species, leaf_tax, level, leaf_angles):
    blocks = []
    start = 0
    for i in range(1, len(leaf_species) + 1):
        if i == len(leaf_species) or leaf_tax[i][level] != leaf_tax[start][level]:
            blocks.append({
                "clade": leaf_tax[start][level],
                "color": leaf_tax[start][level + "_color"],
                "start_idx": start,
                "end_idx": i - 1,
                "start_angle": leaf_angles[start],
                "end_angle": leaf_angles[i - 1],
            })
            start = i
    return blocks


def main():
    tree = Tree(str(CACHE_DIR / "tree.newick"), format=1)
    leaves = tree.get_leaves()
    n = len(leaves)
    leaf_species = [str(l.name) for l in leaves]
    step = (360.0 - GAP_DEG) / (n - 1 if GAP_DEG > 0 else n)
    leaf_angles = [round(START_DEG - i * step, 4) for i in range(n)]
    print(f"{n} leaves")

    # Taxonomy + colours
    ncbi = NCBITaxa()
    tax = leaf_taxonomy(ncbi, leaf_species)
    leaf_tax = []
    for sp in leaf_species:
        info = {"sp": sp, **tax[sp], "cilia": "NR"}
        leaf_tax.append(info)
    for level in LEVEL_RANKS:
        colors = assign_colors([t[level] for t in leaf_tax])
        for t in leaf_tax:
            t[level + "_color"] = colors[t[level]]

    # Nodes: preorder ids, contiguous leaf ranges, depth-based radii
    leaf_idx = {id(l): i for i, l in enumerate(leaves)}
    node_id = {}
    for i, node in enumerate(tree.traverse("preorder")):
        node_id[id(node)] = i
    depth = {id(tree): 0}
    for node in tree.traverse("preorder"):
        for c in node.children:
            depth[id(c)] = depth[id(node)] + 1
    max_depth = max(depth.values()) or 1

    rng = {}
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            li = leaf_idx[id(node)]
            rng[id(node)] = (li, li)
        else:
            rng[id(node)] = (rng[id(node.children[0])][0], rng[id(node.children[-1])][1])

    nodes = []
    for node in tree.traverse("preorder"):
        l0, l1 = rng[id(node)]
        r_phylo = R_ROOT + (R_TREE - R_ROOT) * depth[id(node)] / max_depth
        is_leaf = node.is_leaf()
        nodes.append({
            "id": node_id[id(node)],
            "p": node_id[id(node.up)] if node.up is not None else None,
            "c": [node_id[id(c)] for c in node.children],
            "r": R_TREE if is_leaf else round(r_phylo, 2),
            "r_phylo": round(r_phylo, 2),
            "a": round((leaf_angles[l0] + leaf_angles[l1]) / 2.0, 4),
            "leaf": is_leaf,
            "leaf_idx": l0 if is_leaf else None,
            "sp": leaf_species[l0] if is_leaf else None,
            "l0": l0,
            "l1": l1,
        })

    gene_names = json.loads(DATA_JSON.read_text())["names"]

    layout = {
        "n_leaves": n,
        "leaf_species": leaf_species,
        "leaf_angles": leaf_angles,
        "leaf_taxonomies": leaf_tax,
        "clade_levels": {lv: clade_blocks(leaf_species, leaf_tax, lv, leaf_angles) for lv in LEVEL_RANKS},
        "nodes": nodes,
        "gene_names": gene_names,
        "gap_deg": GAP_DEG,
        "start_deg": START_DEG,
        "r_root": R_ROOT,
        "r_tree": R_TREE,
    }
    layout["clade_blocks"] = layout["clade_levels"]["supergroup"]
    LAYOUT_OUT.write_text(json.dumps(layout, separators=(",", ":")))
    print(f"Wrote {LAYOUT_OUT} ({LAYOUT_OUT.stat().st_size / 1e6:.1f} MB, {len(nodes)} nodes)")

    # Presence bitmap in gene_names x leaf order
    print("Loading presence table...")
    table = pd.read_csv(CACHE_DIR / "table.tsv", sep="\t", index_col=0)
    table.columns = table.columns.astype(str)
    table = table.reindex(columns=leaf_species, fill_value=0)
    table = table.reindex(index=gene_names, fill_value=0)
    missing = int((table.sum(axis=1) == 0).sum())
    bits = np.packbits(table.to_numpy(dtype=np.uint8) > 0, axis=1, bitorder="little")
    assert bits.shape[1] == math.ceil(n / 8)
    with open(PRESENCE_OUT, "wb") as f:
        f.write(b"DLTP")
        f.write(struct.pack("<IH", len(gene_names), n))
        f.write(bits.tobytes())
    print(f"Wrote {PRESENCE_OUT} ({PRESENCE_OUT.stat().st_size / 1e6:.1f} MB; "
          f"{len(gene_names)} genes, {missing} with no presence)")


if __name__ == "__main__":
    main()
