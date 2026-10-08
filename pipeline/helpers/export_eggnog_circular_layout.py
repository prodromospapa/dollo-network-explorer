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
  tree_events_eggnog.bin    Dollo (COUNT) events per gene, as layout node ids:
                            "DLTE" | uint32 n_genes | uint32 offsets[n_genes+1] |
                            uint16 gain_node[n_genes] (0xFFFF = none) |
                            uint16 loss_nodes[...]  (loss = on branch INTO node)

Unary nodes are collapsed so the tree matches COUNT's: COUNT numbers leaves
first (newick order), then internal nodes in postorder.

Nodes carry a contiguous leaf range [l0, l1] instead of an explicit leaf list
(12.5k leaves would make per-node lists tens of MB).

Taxonomy: every leaf carries its name at each rank in TAX_LEVELS (falling back
to the nearest higher rank NCBI defines). Clade blocks and colours are derived
in the browser from these names, so they can follow the current view.

Usage (from repo root):
  python3 pipeline/helpers/export_eggnog_circular_layout.py [--layout-only]
"""

import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ete3 import NCBITaxa, Tree

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "cache_eggnog_full"
LAYOUT_OUT = ROOT / "tree_layout_eggnog.json"
PRESENCE_OUT = ROOT / "tree_presence_eggnog.bin"
DATA_JSON = ROOT / "data_eggnog.json"
EVENTS_OUT = ROOT / "tree_events_eggnog.bin"
HISTORY_BITS = CACHE_DIR / "history_bits.npz"

START_DEG = 90.0
GAP_DEG = 0.0
R_ROOT = 35.0
R_TREE = 300.0

# Taxonomy levels exposed in the UI selector, coarse to fine. Each level falls
# back to the nearest higher rank NCBI defines for that lineage.
TAX_LEVELS = ["domain", "kingdom", "phylum", "class", "order", "family", "genus"]
RANK_ALIASES = {"domain": ["domain", "superkingdom"]}


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
        prev = "Unclassified"
        for level in TAX_LEVELS:
            val = next((by_rank[r] for r in RANK_ALIASES.get(level, [level]) if by_rank.get(r)), None)
            info[level] = prev = val or prev
        out[t] = info
    return out


def main():
    tree = Tree(str(CACHE_DIR / "tree.newick"), format=1)
    for x in [x for x in tree.traverse() if len(x.children) == 1 and x.up is not None]:
        x.delete(prevent_nondicotomic=False, preserve_branch_length=True)
    leaves = tree.get_leaves()
    n = len(leaves)
    leaf_species = [str(l.name) for l in leaves]
    step = (360.0 - GAP_DEG) / (n - 1 if GAP_DEG > 0 else n)
    leaf_angles = [round(START_DEG - i * step, 4) for i in range(n)]
    print(f"{n} leaves")

    # Taxonomy + colours
    ncbi = NCBITaxa()
    tax = leaf_taxonomy(ncbi, leaf_species)
    leaf_tax = [{"sp": sp, **tax[sp]} for sp in leaf_species]

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

    # COUNT node index: leaves 0..n-1, then internal nodes in postorder.
    count_idx = {id(l): i for i, l in enumerate(leaves)}
    for node in tree.traverse("postorder"):
        if not node.is_leaf():
            count_idx[id(node)] = len(count_idx)
    internal_ids = [int(x.name) for x in tree.traverse() if not x.is_leaf() and str(x.name).isdigit()]
    clade_names = ncbi.get_taxid_translator(internal_ids)

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
            "ci": count_idx[id(node)],
            "name": None if is_leaf else clade_names.get(int(node.name)) if str(node.name).isdigit() else None,
        })

    gene_names = json.loads(DATA_JSON.read_text())["names"]

    layout = {
        "n_leaves": n,
        "leaf_species": leaf_species,
        "leaf_angles": leaf_angles,
        "leaf_taxonomies": leaf_tax,
        "tax_levels": TAX_LEVELS,
        "nodes": nodes,
        "gene_names": gene_names,
        "gap_deg": GAP_DEG,
        "start_deg": START_DEG,
        "r_root": R_ROOT,
        "r_tree": R_TREE,
    }
    LAYOUT_OUT.write_text(json.dumps(layout, separators=(",", ":")))
    print(f"Wrote {LAYOUT_OUT} ({LAYOUT_OUT.stat().st_size / 1e6:.1f} MB, {len(nodes)} nodes)")

    if "--layout-only" in sys.argv:
        return
    write_events(gene_names, nodes)

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


def write_events(gene_names, nodes):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from count_bits import load_npz, unpack

    hist_genes, n_nodes, bits = load_npz(str(HISTORY_BITS))
    if n_nodes != len(nodes):
        raise SystemExit(f"COUNT has {n_nodes} nodes, layout has {len(nodes)}")
    ci_to_layout = np.empty(n_nodes, dtype=np.int64)
    for nd in nodes:
        ci_to_layout[nd["ci"]] = nd["id"]
    row = {g: i for i, g in enumerate(hist_genes)}
    loss = unpack(bits["loss"], n_nodes).astype(bool)
    gain = unpack(bits["gain"], n_nodes).astype(bool)

    offsets, gains, flat = [0], [], []
    for g in gene_names:
        i = row.get(g)
        lost = [] if i is None else sorted(int(x) for x in ci_to_layout[np.flatnonzero(loss[i])])
        gn = [] if i is None else ci_to_layout[np.flatnonzero(gain[i])]
        gains.append(int(gn[0]) if len(gn) else 0xFFFF)
        flat.extend(lost)
        offsets.append(len(flat))
    with open(EVENTS_OUT, "wb") as f:
        f.write(b"DLTE")
        f.write(struct.pack("<I", len(gene_names)))
        f.write(np.asarray(offsets, dtype="<u4").tobytes())
        f.write(np.asarray(gains, dtype="<u2").tobytes())
        f.write(np.asarray(flat, dtype="<u2").tobytes())
    print(f"Wrote {EVENTS_OUT} ({EVENTS_OUT.stat().st_size / 1e6:.1f} MB; {len(flat)} losses)")


if __name__ == "__main__":
    main()
