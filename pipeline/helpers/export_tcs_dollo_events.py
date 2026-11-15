#!/usr/bin/env python3
"""
Dollo (COUNT) events for the 196-species TCS tree, in the same binary format
the EggNOG tree uses, so the explorer's tree view can draw the loss dots and
gain rings for TCS too.

Inputs (repo root):
  tree_layout.json                 TCS species tree (topology, leaf order, gene names)
  tree_presence.bin                orthogroup presence, DLTP (genes x leaves)
  tree_presence.ortholog.bin       ortholog presence, DLTP (same gene/leaf order)
Outputs (repo root):
  tree_events.bin                  orthogroup dataset, DLTE
  tree_events.ortholog.bin         ortholog dataset, DLTE
Work files: cache_tcs_full/<dataset>/ (table.tsv, tree.newick, history_bits.npz)

The tree is written as newick from the layout's own nodes, so COUNT's leaves
are the layout's leaves. COUNT numbers leaves first (newick order), then
internal nodes in postorder; that mapping is verified against the presence
matrix before any event is written, as export_eggnog_circular_layout.py does.

Gain/loss rule: gain penalty 1000, loss penalty 1 (single origin, as N2_reconstruct.sh).

Usage (repo root):
  python3 pipeline/helpers/export_tcs_dollo_events.py [--dataset orthogroup|ortholog] [--skip-count]
"""
import argparse
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from count_bits import load_npz, unpack  # noqa: E402

LAYOUT = ROOT / "tree_layout.json"
DATASETS = {
    "orthogroup": ("tree_presence.bin", "tree_events.bin"),
    "ortholog": ("tree_presence.ortholog.bin", "tree_events.ortholog.bin"),
}


def read_presence(path, n_genes, n_leaves):
    """DLTP -> bool matrix (genes x leaves, leaf order = layout leaf_idx)."""
    raw = Path(path).read_bytes()
    assert raw[:4] == b"DLTP", path
    g, l = struct.unpack("<IH", raw[4:10])
    assert (g, l) == (n_genes, n_leaves), (path, g, l)
    width = -(-n_leaves // 8)
    bits = np.frombuffer(raw, np.uint8, offset=10).reshape(n_genes, width)
    return np.unpackbits(bits, axis=1, bitorder="little")[:, :n_leaves].astype(bool)


def newick_and_count_index(nodes_by_id, root=0):
    """Newick text (children in layout order) and COUNT's node numbering:
    leaves 0..n-1 in newick order, internal nodes after, in postorder."""
    leaf_order, ci = [], {}

    def walk_text(nid):
        nd = nodes_by_id[nid]
        if nd["leaf"]:
            leaf_order.append(nid)
            return nd["sp"]
        return "(" + ",".join(walk_text(c) for c in nd["c"]) + ")"

    text = walk_text(root) + ";"
    for nid in leaf_order:
        ci[nid] = len(ci)

    def post(nid):
        nd = nodes_by_id[nid]
        if not nd["leaf"]:
            for c in nd["c"]:
                post(c)
            ci[nid] = len(ci)

    post(root)
    return text, ci


def add_layout_ranges(layout_path, nodes_by_id, ci):
    """Give every node the fields the explorer's tree view reads: l0/l1 (leaf
    range, leaves are contiguous in layout order), ci (COUNT node index), name."""
    layout = json.loads(layout_path.read_text())
    for nd in layout["nodes"]:
        leaves = sorted(nd["leaves"])
        assert leaves == list(range(leaves[0], leaves[-1] + 1)), f"node {nd['id']} leaves not contiguous"
        nd["l0"], nd["l1"] = leaves[0], leaves[-1]
        nd["ci"] = ci[nd["id"]]
        nd.setdefault("name", None)
    layout_path.write_text(json.dumps(layout))
    print(f"Added l0/l1/ci/name to {len(layout['nodes'])} nodes in {layout_path.name}")


def run_count(ds_dir, table, tree):
    """COUNT via parallel_count.py (resumes from finished chunks)."""
    out = ds_dir / "history_bits.npz"
    if not out.exists():
        env = dict(os.environ)
        env.setdefault("COUNT_WORKERS", "32")
        env.setdefault("COUNT_CHUNKS", "32")
        subprocess.run([sys.executable, str(ROOT / "pipeline/helpers/parallel_count.py"),
                        str(table), str(tree), str(out)], check=True, cwd=ROOT, env=env)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), action="append")
    ap.add_argument("--skip-count", action="store_true", help="reuse existing history_bits.npz")
    a = ap.parse_args()
    datasets = a.dataset or list(DATASETS)

    layout = json.loads(LAYOUT.read_text())
    nodes = layout["nodes"]
    by_id = {n["id"]: n for n in nodes}
    leaves = layout["leaf_species"]
    genes = layout["gene_names"]
    n_leaves, n_genes = layout["n_leaves"], len(genes)
    newick, ci = newick_and_count_index(by_id)
    add_layout_ranges(LAYOUT, by_id, ci)
    n_nodes = len(nodes)
    assert len(ci) == n_nodes, "layout is not a single rooted tree"
    ci_to_layout = np.empty(n_nodes, dtype=np.int64)
    for nid, c in ci.items():
        ci_to_layout[c] = nid

    for ds in datasets:
        presence_file, events_file = DATASETS[ds]
        ds_dir = ROOT / "cache_tcs_full" / ds
        ds_dir.mkdir(parents=True, exist_ok=True)
        pres = read_presence(ROOT / presence_file, n_genes, n_leaves)

        # COUNT inputs: table with leaf names as columns (in layout leaf order)
        table = ds_dir / "table.tsv"
        tree = ds_dir / "tree.newick"
        tree.write_text(newick + "\n")
        with open(table, "w") as fh:
            fh.write("\t".join(["gene"] + leaves) + "\n")
            for g, row in zip(genes, pres):
                fh.write(g + "\t" + "\t".join("1" if x else "0" for x in row) + "\n")

        if a.skip_count and (ds_dir / "history_bits.npz").exists():
            hist = ds_dir / "history_bits.npz"
        else:
            hist = run_count(ds_dir, table, tree)

        hist_genes, hn, bits = load_npz(str(hist))
        assert hn == n_nodes, f"COUNT has {hn} nodes, layout has {n_nodes}"
        row = {g: i for i, g in enumerate(hist_genes)}
        loss = unpack(bits["loss"], n_nodes).astype(bool)
        gain = unpack(bits["gain"], n_nodes).astype(bool)
        presence = unpack(bits["presence"], n_nodes).astype(bool)

        # Verify: COUNT's presence at each leaf equals the input matrix
        leaf_ci = [ci[nid] for nid in range(n_nodes) if by_id[nid]["leaf"]]
        leaf_idx = [by_id[nid]["leaf_idx"] for nid in range(n_nodes) if by_id[nid]["leaf"]]
        idx_rows = [row[g] for g in genes]
        if not np.array_equal(presence[np.ix_(idx_rows, leaf_ci)], pres[:, leaf_idx]):
            sys.exit(f"{ds}: COUNT leaf presence does not match the input matrix; mapping is off")
        print(f"{ds}: COUNT leaf order verified ({n_leaves} leaves, {n_genes} genes)")

        offsets, gains, flat = [0], [], []
        for g in genes:
            i = row[g]
            lost = sorted(int(x) for x in ci_to_layout[np.flatnonzero(loss[i])])
            gn = ci_to_layout[np.flatnonzero(gain[i])]
            gains.append(int(gn[0]) if len(gn) else 0xFFFF)
            flat.extend(lost)
            offsets.append(len(flat))
        out = ROOT / events_file
        with open(out, "wb") as f:
            f.write(b"DLTE")
            f.write(struct.pack("<I", n_genes))
            f.write(np.asarray(offsets, dtype="<u4").tobytes())
            f.write(np.asarray(gains, dtype="<u2").tobytes())
            f.write(np.asarray(flat, dtype="<u2").tobytes())
        print(f"{ds}: wrote {out.name} ({len(flat)} losses, "
              f"{sum(1 for x in gains if x != 0xFFFF)} genes with a gain)")


if __name__ == "__main__":
    main()
