#!/usr/bin/env python3
"""Leiden community detection on the gene Jaccard-similarity graph.

Reads the precomputed Jaccard matrix (from jaccard_matrix.py) and gene index,
builds an igraph weighted graph (edges above a configurable floor), then runs
Leiden community detection (ModularityVertexPartition) to identify gene modules.

Outputs:
    leiden_clusters.tsv   -- gene_name, cluster_id, loss_count  (one row per gene)
    leiden_summary.tsv    -- cluster_id, size, n_panel, panel_pct, top_members

Usage:
    python3 leiden_cluster.py [--resolution 1.0] [--jaccard-floor 0.05] [--min-loss 5]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import igraph as ig
    import leidenalg
except ImportError:
    sys.exit(
        "ERROR: leidenalg and python-igraph are required.\n"
        "  pip install leidenalg python-igraph"
    )


def load_gene_index(path):
    """Return list of dicts with idx, fid, name."""
    genes = []
    with open(path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            genes.append({"idx": int(parts[0]), "fid": parts[1], "name": parts[2]})
    return genes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resolution", type=float, default=1.0,
                    help="Leiden resolution parameter (default: 1.0). "
                         "Higher = more / smaller clusters.")
    ap.add_argument("--jaccard-floor", type=float, default=0.05,
                    help="Minimum Jaccard similarity to create an edge (default: 0.05)")
    ap.add_argument("--min-loss", type=int, default=5,
                    help="Minimum loss events for a gene to be included (default: 5)")
    ap.add_argument("--matrix", type=str, default=None,
                    help="Path to jaccard_matrix.npy")
    ap.add_argument("--index", type=str, default=None,
                    help="Path to jaccard_genes.tsv")
    ap.add_argument("--loss-counts", type=str, default=None,
                    help="Path to loss_counts.npy")
    ap.add_argument("--panel", type=str, default=None,
                    help="Path to cilia panel CSV (optional, for enrichment stats)")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="Output directory (default: ../results/)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility")
    args = ap.parse_args()

    # Resolve paths relative to this script
    base = Path(__file__).resolve().parent.parent
    results = base / "results"

    matrix_path = args.matrix or str(results / "jaccard_matrix.npy")
    index_path = args.index or str(results / "jaccard_genes.tsv")
    loss_path = args.loss_counts or str(results / "loss_counts.npy")
    out_dir = Path(args.out_dir) if args.out_dir else results

    # Load panel genes if available
    panel_genes = set()
    panel_path = args.panel
    if panel_path is None:
        # Try common locations
        for candidate in [base.parent / "ciliary_genes.csv",
                          base / "ciliary_genes.csv",
                          base.parent / "cilia_gene_panel_scgsv1.csv",
                          base / "cilia_gene_panel_scgsv1.csv"]:
            if candidate.exists():
                panel_path = str(candidate)
                break
    if panel_path and Path(panel_path).exists():
        import csv
        with open(panel_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                sym = (row.get("Gene Name") or row.get("resolved_symbol") or "").strip()
                if sym:
                    panel_genes.add(sym)
        print(f"Loaded {len(panel_genes)} panel genes", file=sys.stderr)

    # Load data
    print("Loading Jaccard matrix...", file=sys.stderr)
    J = np.load(matrix_path, mmap_mode="r")
    genes = load_gene_index(index_path)
    loss_counts = np.load(loss_path)

    n_total = len(genes)
    print(f"  {n_total} genes, matrix shape {J.shape}", file=sys.stderr)

    # Filter by minimum losses
    eligible_mask = loss_counts >= args.min_loss
    eligible_indices = [g["idx"] for g in genes if eligible_mask[g["idx"]]]
    eligible_names = [g["name"] for g in genes if eligible_mask[g["idx"]]]
    n_eligible = len(eligible_indices)
    print(f"  {n_eligible} genes with >= {args.min_loss} losses", file=sys.stderr)

    # Build submatrix for eligible genes
    print("Extracting submatrix for eligible genes...", file=sys.stderr)
    idx_arr = np.array(eligible_indices)
    # Read the relevant rows/cols from the (potentially mmap'd) full matrix
    sub_J = np.array(J[np.ix_(idx_arr, idx_arr)], dtype=np.float32)
    eligible_losses = loss_counts[idx_arr]

    # Build igraph from the upper triangle of the submatrix
    print(f"Building graph (Jaccard >= {args.jaccard_floor})...", file=sys.stderr)
    rows_i, cols_i = np.triu_indices(n_eligible, k=1)
    weights = sub_J[rows_i, cols_i]

    mask = weights >= args.jaccard_floor
    edge_rows = rows_i[mask]
    edge_cols = cols_i[mask]
    edge_weights = weights[mask]

    n_edges = len(edge_weights)
    print(f"  {n_edges} edges above floor", file=sys.stderr)

    g = ig.Graph(n=n_eligible, edges=list(zip(edge_rows.tolist(), edge_cols.tolist())),
                 directed=False)
    g.es["weight"] = edge_weights.tolist()
    g.vs["name"] = eligible_names
    g.vs["losses"] = eligible_losses.tolist()

    # Run Leiden
    print(f"Running Leiden (resolution={args.resolution}, seed={args.seed})...",
          file=sys.stderr)
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=args.resolution,
        seed=args.seed,
    )

    n_clusters = len(partition)
    modularity = partition.modularity
    print(f"  Found {n_clusters} clusters (modularity={modularity:.4f})", file=sys.stderr)

    # Build output
    cluster_ids = partition.membership
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-gene output
    clusters_path = out_dir / "leiden_clusters.tsv"
    with open(clusters_path, "w") as f:
        f.write("gene_name\tcluster_id\tloss_count\n")
        for i, name in enumerate(eligible_names):
            f.write(f"{name}\t{cluster_ids[i]}\t{int(eligible_losses[i])}\n")
    print(f"  Written: {clusters_path}", file=sys.stderr)

    # Summary output
    from collections import defaultdict
    cluster_members = defaultdict(list)
    cluster_losses = defaultdict(list)
    for i, name in enumerate(eligible_names):
        cid = cluster_ids[i]
        cluster_members[cid].append(name)
        cluster_losses[cid].append(int(eligible_losses[i]))

    summary_path = out_dir / "leiden_summary.tsv"
    with open(summary_path, "w") as f:
        f.write("cluster_id\tsize\tn_panel\tpanel_pct\tmean_losses\ttop_members\n")
        for cid in sorted(cluster_members.keys()):
            members = cluster_members[cid]
            size = len(members)
            n_panel = sum(1 for m in members if m in panel_genes)
            panel_pct = n_panel / size * 100 if size > 0 else 0.0
            mean_loss = np.mean(cluster_losses[cid])
            # Sort members by loss count descending for the sample
            member_losses = list(zip(members, cluster_losses[cid]))
            member_losses.sort(key=lambda x: -x[1])
            top = ";".join(m for m, _ in member_losses[:20])
            f.write(f"{cid}\t{size}\t{n_panel}\t{panel_pct:.1f}\t{mean_loss:.1f}\t{top}\n")
    print(f"  Written: {summary_path}", file=sys.stderr)

    # Print summary
    sizes = [len(cluster_members[c]) for c in sorted(cluster_members.keys())]
    print(f"\nDone.", file=sys.stderr)
    print(f"  Clusters: {n_clusters}", file=sys.stderr)
    print(f"  Modularity: {modularity:.4f}", file=sys.stderr)
    print(f"  Cluster sizes: min={min(sizes)}, max={max(sizes)}, "
          f"median={np.median(sizes):.0f}, mean={np.mean(sizes):.1f}", file=sys.stderr)
    if panel_genes:
        total_panel_in = sum(1 for name in eligible_names if name in panel_genes)
        print(f"  Panel genes in eligible set: {total_panel_in}/{len(panel_genes)}",
              file=sys.stderr)


if __name__ == "__main__":
    main()

