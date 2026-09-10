#!/usr/bin/env python3
"""
Build a self-contained HTML network explorer from the Jaccard matrix.

Embeds top-K partners per gene as inline JSON, uses Cytoscape.js for
interactive network rendering. Open the output HTML in any modern browser.

Usage:
    python build_html_explorer.py [--top-k 50] [--min-loss 5] [--jaccard-floor 0.05]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Build interactive HTML network explorer")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Max partners to store per gene (default: 50)")
    parser.add_argument("--min-loss", type=int, default=5,
                        help="Minimum losses for a gene to be included (default: 5)")
    parser.add_argument("--jaccard-floor", type=float, default=0.05,
                        help="Minimum Jaccard to include as partner (default: 0.05)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output HTML file path (default: ./index.html, alongside this script)")
    args = parser.parse_args()

    website_dir = Path(__file__).resolve().parent
    base = website_dir.parent
    results = base / "results"

    print("Loading data...")
    J = np.load(results / "jaccard_matrix.npy", mmap_mode="r")
    loss_counts = np.load(results / "loss_counts.npy")

    genes = []
    with open(results / "jaccard_genes.tsv") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            genes.append({"idx": int(parts[0]), "fid": parts[1], "name": parts[2]})

    n = len(genes)
    print(f"  {n} genes, matrix shape {J.shape}")

    # Filter genes with enough losses
    eligible = [g for g in genes if loss_counts[g["idx"]] >= args.min_loss]
    print(f"  {len(eligible)} genes with >= {args.min_loss} losses")

    # Build name -> idx lookup
    name_to_idx = {g["name"]: g["idx"] for g in genes}

    # Build per-gene top-K partner lists
    print(f"Building top-{args.top_k} partner lists (jaccard >= {args.jaccard_floor})...")
    gene_data = {}  # name -> {losses, partners: [{name, jaccard, losses}]}

    eligible_indices = sorted([g["idx"] for g in eligible])
    idx_to_name = {g["idx"]: g["name"] for g in genes}

    total = len(eligible_indices)
    for progress, idx in enumerate(eligible_indices):
        if progress % 1000 == 0:
            print(f"  {progress}/{total}...")

        row = np.array(J[idx, :], dtype=np.float32)  # force read from mmap
        name = idx_to_name[idx]
        losses = int(loss_counts[idx])

        # Zero out self
        row[idx] = 0.0

        # Only consider eligible partners
        # Get top-K indices above floor
        above_floor = np.where(row >= args.jaccard_floor)[0]
        if len(above_floor) == 0:
            gene_data[name] = {"l": losses, "p": []}
            continue

        scores = row[above_floor]
        if len(scores) > args.top_k:
            topk_local = np.argpartition(scores, -args.top_k)[-args.top_k:]
            topk_indices = above_floor[topk_local]
            topk_scores = scores[topk_local]
        else:
            topk_indices = above_floor
            topk_scores = scores

        # Sort by score descending
        order = np.argsort(-topk_scores)
        partners = []
        for o in order:
            pidx = int(topk_indices[o])
            pname = idx_to_name.get(pidx)
            if pname is None:
                continue
            partners.append({
                "n": pname,
                "j": round(float(topk_scores[o]), 6),
                "l": int(loss_counts[pidx])
            })

        gene_data[name] = {"l": losses, "p": partners}

    print(f"  Built data for {len(gene_data)} genes")

    # All gene names for autocomplete
    all_names = sorted(gene_data.keys())

    # Load curated ciliary gene sets for the cluster-view and network filters.
    data_roots = [base, base.parent]
    ciliary_genes_path = next((root / "ciliary_genes.csv"
                               for root in data_roots
                               if (root / "ciliary_genes.csv").exists()), None)
    ciliacarta_path = next((root / "CiliaCarta.csv"
                            for root in data_roots
                            if (root / "CiliaCarta.csv").exists()), None)

    # 1. Load CiliaCarta strictly from CiliaCarta.csv (the canonical 935-gene compendium)
    cc_genes = set()
    if ciliacarta_path and ciliacarta_path.exists():
        with open(ciliacarta_path, newline="") as f:
            for row in csv.DictReader(f):
                sym = (row.get("Associated Gene Name") or "").strip()
                if sym:
                    cc_genes.add(sym)

    # 2. Load SYSCILIA gold standards and localizations from ciliary_genes.csv
    v2_genes = set()
    v1_genes = set()
    all_curated = set()
    cilia_info = {}

    if ciliary_genes_path and ciliary_genes_path.exists():
        with open(ciliary_genes_path, newline="") as f:
            for row in csv.DictReader(f):
                gene = (row.get("Gene Name") or "").strip()
                if not gene:
                    continue
                all_curated.add(gene)
                is_first = (row.get("First order") or "").strip().lower() == "x"
                is_second = (row.get("Second order") or "").strip().lower() == "x"
                is_v1 = (row.get("In SCGSv1") or "").strip().lower() == "x" or (row.get("Predicted in SCGSv1 paper") or "").strip().lower() == "x"
                loc = (row.get("Localisation") or "").strip()

                if is_first or is_second:
                    v2_genes.add(gene)
                if is_v1:
                    v1_genes.add(gene)

                v2_order = "First order" if is_first else ("Second order" if is_second else "")
                cilia_info[gene] = {
                    "v2": v2_order,
                    "v1": is_v1,
                    "cc": gene in cc_genes,
                    "loc": loc
                }

    # Register any CiliaCarta genes not present in ciliary_genes.csv
    for gene in cc_genes:
        if gene not in cilia_info:
            cilia_info[gene] = {
                "v2": "",
                "v1": False,
                "cc": True,
                "loc": ""
            }

    all_ciliary = all_curated | cc_genes

    ciliary_sets = {
        "syscilia_v2": sorted(v2_genes),
        "syscilia_v1": sorted(v1_genes),
        "ciliacarta": sorted(cc_genes),
        "ciliacarta_full": sorted(cc_genes),
        "both": sorted(all_ciliary),
        "all_ciliary": sorted(all_ciliary),
        "syscilia": sorted(v2_genes),
    }
    print(f"  Ciliary filter sets: "
          f"SCGSv2={len(v2_genes)}, SCGSv1={len(v1_genes)}, CiliaCarta={len(cc_genes)}, "
          f"All Ciliary (Union)={len(all_ciliary)}")

    # Load Leiden cluster assignments (if available)
    cluster_data = {}   # cluster_id (int) -> [gene_name, ...]
    gene_to_cluster = {}  # gene_name -> cluster_id
    leiden_path = results / "leiden_clusters.tsv"
    if leiden_path.exists():
        print("Loading Leiden cluster assignments...")
        from collections import defaultdict
        clusters_raw = defaultdict(list)
        with open(leiden_path) as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    gname, cid = parts[0], int(parts[1])
                    if gname in gene_data:
                        clusters_raw[cid].append(gname)
                        gene_to_cluster[gname] = cid
        cluster_data = dict(clusters_raw)
        print(f"  {len(cluster_data)} clusters, {len(gene_to_cluster)} genes assigned")
    else:
        print("No Leiden clusters found (leiden_clusters.tsv missing). "
              "All-clusters view will be disabled.")

    # Annotate clusters with GO terms
    cluster_names = {}
    if cluster_data:
        cluster_names = annotate_clusters_with_go(cluster_data, base)
        print(f"  Annotated {len(cluster_names)} clusters with top enriched terms")

        # Also update leiden_summary.tsv with cluster names if it exists
        summary_path = results / "leiden_summary.tsv"
        if summary_path.exists():
            try:
                import pandas as pd
                sdf = pd.read_csv(summary_path, sep="\t")
                sdf["cluster_name"] = sdf["cluster_id"].map(lambda cid: cluster_names.get(int(cid), f"Cluster {cid}"))
                cols = list(sdf.columns)
                if "cluster_name" in cols:
                    cols.remove("cluster_name")
                    cols.insert(1, "cluster_name")
                    sdf = sdf[cols]
                sdf.to_csv(summary_path, sep="\t", index=False)
                print(f"  Updated {summary_path} with cluster_name column")
            except Exception as e:
                print(f"  Note: could not update summary TSV: {e}")

    # Build the HTML
    print("Generating HTML...")

    output_path = args.output or str(website_dir / "index.html")

    html = build_html(gene_data, all_names, args.top_k, cluster_data,
                      gene_to_cluster, cluster_names, ciliary_sets, cilia_info)

    with open(output_path, "w") as f:
        f.write(html)

    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Written to {output_path} ({size_mb:.1f} MB)")

def annotate_clusters_with_go(cluster_data, base_dir):
    """Annotate each cluster with its top statistically enriched GO terms."""
    import pickle
    from collections import Counter
    import scipy.stats as stats

    candidates = [
        base_dir / "data" / "go",
        base_dir.parent / "junk" / "coevolution_framework" / "helpers" / "go_data",
        base_dir / "junk" / "coevolution_framework" / "helpers" / "go_data",
    ]
    go_dir = None
    for c in candidates:
        if (c / "symbol_go_map.pkl").exists() and (c / "go-basic.obo").exists():
            go_dir = c
            break

    if not go_dir:
        print("  GO annotation data not found. Using numeric cluster IDs.")
        return {cid: f"Cluster {cid}" for cid in cluster_data}

    print(f"  Loading GO data from {go_dir}...")
    with open(go_dir / "symbol_go_map.pkl", "rb") as f:
        sgm = pickle.load(f)

    go_names = {}
    with open(go_dir / "go-basic.obo") as f:
        cur_id, cur_name = None, None
        for line in f:
            if line.startswith("[Term]"):
                if cur_id and cur_name:
                    go_names[cur_id] = cur_name
                cur_id, cur_name = None, None
            elif line.startswith("id: GO:"):
                cur_id = line.strip().split()[1]
            elif line.startswith("name: "):
                cur_name = line.strip()[6:]
        if cur_id and cur_name:
            go_names[cur_id] = cur_name

    all_genes = set(g for members in cluster_data.values() for g in members)
    bg_counts = Counter()
    for g in all_genes:
        for go in sgm.get(g, []):
            bg_counts[go] += 1
    N_total = len(all_genes)

    blacklist = {
        "protein binding", "cytoplasm", "nucleus", "cytosol", "membrane", "nucleoplasm",
        "cellular component", "biological_process", "molecular_function", "intracellular",
        "cell", "organelle", "intracellular organelle", "binding", "catalytic activity",
        "metabolic process", "cellular metabolic process", "cellular process",
        "metal ion binding", "ion binding", "protein-containing complex"
    }

    cluster_names = {}
    for cid, genes in cluster_data.items():
        K = len(genes)
        c_counts = Counter()
        for g in genes:
            for go in sgm.get(g, []):
                c_counts[go] += 1

        scored = []
        for go, k in c_counts.items():
            term = go_names.get(go)
            if not term or term.lower() in blacklist:
                continue
            M = bg_counts[go]
            if M < 2 or M > 2500:
                continue
            table = [[k, K - k], [M - k, N_total - K - (M - k)]]
            odds, pval = stats.fisher_exact(table, alternative="greater")
            if pval < 0.05:
                scored.append((pval, -k, term))

        scored.sort()
        if scored:
            picked = []
            seen_words = set()
            for pval, neg_k, term in scored:
                words = set(term.lower().split())
                if len(words & seen_words) >= min(3, len(words)):
                    continue
                seen_words.update(words)
                t = term
                if len(t) > 38:
                    t = t[:36] + "…"
                t = t[0].upper() + t[1:]
                picked.append(t)
                if len(picked) == 2:
                    break
            cluster_names[cid] = " / ".join(picked)
        else:
            cluster_names[cid] = f"Module {cid} (" + ", ".join(sorted(genes)[:3]) + ")"

    return cluster_names


def build_html(gene_data, all_names, top_k, cluster_data=None, gene_to_cluster=None,
               cluster_names=None, ciliary_sets=None, cilia_info=None):
    """Build the complete self-contained HTML string."""

    data_json = json.dumps(gene_data, separators=(",", ":"))
    names_json = json.dumps(all_names, separators=(",", ":"))

    # Cluster data for the all-clusters view
    if cluster_data:
        clusters_json = json.dumps(cluster_data, separators=(",", ":"))
        gene_cluster_json = json.dumps(gene_to_cluster or {}, separators=(",", ":"))
        cluster_names_json = json.dumps(cluster_names or {}, separators=(",", ":"))
    else:
        clusters_json = "{}"
        gene_cluster_json = "{}"
        cluster_names_json = "{}"
    ciliary_sets_json = json.dumps(ciliary_sets or {}, separators=(",", ":"))
    cilia_info_json = json.dumps(cilia_info or {}, separators=(",", ":"))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Gene Loss-Concordance Network Explorer</title>
<script src="https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0a0e17;
    color: #e0e6f0;
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}}
#header {{
    background: linear-gradient(135deg, #141a2e 0%, #1a2040 100%);
    border-bottom: 1px solid #2a3050;
    padding: 10px 20px;
    display: flex;
    align-items: center;
    gap: 15px;
    flex-wrap: wrap;
    z-index: 10;
}}
#header h1 {{
    font-size: 16px;
    font-weight: 600;
    color: #7eb8ff;
    white-space: nowrap;
}}
.search-box {{
    position: relative;
    flex: 0 0 280px;
}}
.search-box input {{
    width: 100%;
    padding: 7px 12px;
    border: 1px solid #3a4570;
    border-radius: 6px;
    background: #0d1220;
    color: #e0e6f0;
    font-size: 14px;
    outline: none;
}}
.search-box input:focus {{ border-color: #5a8eff; }}
.search-box input::placeholder {{ color: #556; }}
#suggestions {{
    position: absolute;
    top: 100%;
    left: 0;
    right: 0;
    background: #1a2040;
    border: 1px solid #3a4570;
    border-top: none;
    border-radius: 0 0 6px 6px;
    max-height: 250px;
    overflow-y: auto;
    display: none;
    z-index: 100;
}}
#suggestions div {{
    padding: 6px 12px;
    cursor: pointer;
    font-size: 13px;
}}
#suggestions div:hover, #suggestions div.active {{
    background: #2a3a6a;
}}
.controls {{
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}}
.controls label {{
    font-size: 12px;
    color: #8892b0;
}}
.controls input[type="range"] {{
    width: 120px;
    accent-color: #5a8eff;
}}
.btn {{
    padding: 6px 14px;
    border: 1px solid #3a4570;
    border-radius: 5px;
    background: #1a2040;
    color: #a0b0d0;
    cursor: pointer;
    font-size: 12px;
    transition: all 0.15s;
}}
.btn:hover {{ background: #2a3a6a; color: #fff; }}
.btn-accent {{
    background: #1a3a6a;
    border-color: #3a6aaa;
    color: #7eb8ff;
}}
.btn-accent:hover {{ background: #2a4a8a; }}
.btn-danger {{
    border-color: #6a3a3a;
    color: #ff7e7e;
}}
.btn-danger:hover {{ background: #3a1a1a; }}
.btn-clusters {{
    background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
    border: 1px solid #38bdf8;
    color: #ffffff;
    font-weight: 600;
    cursor: pointer;
    box-shadow: 0 2px 8px rgba(2, 132, 199, 0.4);
}}
.btn-clusters:hover {{
    background: linear-gradient(135deg, #0369a1 0%, #075985 100%);
    color: #ffffff;
    box-shadow: 0 3px 12px rgba(56, 189, 248, 0.6);
}}
.btn-clusters-header {{
    background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
    border: 1px solid #38bdf8;
    color: #ffffff;
    font-weight: 600;
    font-size: 13px;
    padding: 7px 15px;
    border-radius: 6px;
    box-shadow: 0 2px 10px rgba(2, 132, 199, 0.45);
    white-space: nowrap;
    cursor: pointer;
    transition: all 0.2s;
}}
.btn-clusters-header:hover {{
    background: linear-gradient(135deg, #0369a1 0%, #075985 100%);
    box-shadow: 0 4px 14px rgba(56, 189, 248, 0.65);
    transform: translateY(-1px);
}}
.cluster-entry {{
    display: flex;
    align-items: center;
    padding: 7px 12px;
    border-bottom: 1px solid #1a2040;
    cursor: pointer;
    transition: background 0.1s;
    gap: 8px;
}}
.cluster-entry:hover {{ background: #1a2a4a; }}
.cluster-entry .swatch {{
    width: 14px;
    height: 14px;
    border-radius: 3px;
    flex-shrink: 0;
    border: 1px solid rgba(255,255,255,0.15);
}}
.cluster-entry .cname {{
    flex: 1;
    font-size: 13px;
    font-weight: 500;
    color: #e0e6f0;
}}
.cluster-entry .csize {{
    font-size: 11px;
    color: #8892b0;
    flex-shrink: 0;
}}
#main {{
    display: flex;
    flex: 1;
    overflow: hidden;
}}
#cy {{
    flex: 1;
    background: #0a0e17;
    position: relative;
}}
#sidebar {{
    width: 370px;
    background: #111827;
    border-left: 1px solid #2a3050;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    flex-shrink: 0;
}}
#gene-info {{
    padding: 12px;
    border-bottom: 1px solid #2a3050;
    min-height: 70px;
}}
#gene-info h2 {{
    font-size: 18px;
    color: #7eb8ff;
    margin-bottom: 4px;
}}
#gene-info .meta {{
    font-size: 12px;
    color: #8892b0;
}}
#partner-list {{
    flex: 1;
    overflow-y: auto;
    padding: 0;
}}
#partner-list .partner {{
    display: flex;
    align-items: center;
    padding: 6px 10px;
    border-bottom: 1px solid #1a2040;
    cursor: pointer;
    transition: background 0.12s;
    gap: 6px;
    min-width: 0;
}}
#partner-list .partner:hover {{ background: #1a2a4a; }}
#partner-list .partner .rank {{
    color: #64748b;
    font-size: 11px;
    width: 22px;
    text-align: right;
    flex-shrink: 0;
}}
#partner-list .partner .pname {{
    flex: 1;
    min-width: 0;
    font-size: 12.5px;
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}
#partner-list .partner .pname.in-graph {{ color: #5eff8a; }}
#partner-list .partner .jaccard-bar {{
    width: 60px;
    height: 11px;
    background: #1a2040;
    border-radius: 3px;
    overflow: hidden;
    flex-shrink: 0;
    position: relative;
}}
#partner-list .partner .jaccard-bar .fill {{
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s;
}}
#partner-list .partner .jaccard-val {{
    font-size: 11px;
    color: #8892b0;
    width: 38px;
    text-align: right;
    flex-shrink: 0;
}}
#partner-list .partner .ploss {{
    font-size: 11px;
    color: #94a3b8;
    width: 28px;
    text-align: right;
    flex-shrink: 0;
}}
#partner-list .partner .partner-ext-link {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    color: #38bdf8;
    background: rgba(56, 189, 248, 0.1);
    border: 1px solid rgba(56, 189, 248, 0.28);
    font-size: 11px;
    font-weight: 600;
    padding: 1px 5px;
    border-radius: 3px;
    text-decoration: none;
    transition: all 0.15s;
    opacity: 0.9;
    flex-shrink: 0;
    line-height: 1.2;
    margin-left: 2px;
}}
#partner-list .partner:hover .partner-ext-link,
#partner-list .partner .partner-ext-link:hover {{
    opacity: 1;
    background: rgba(56, 189, 248, 0.25);
    border-color: #38bdf8;
    color: #ffffff !important;
    text-decoration: none;
}}
.gene-ext-link {{
    display: inline-flex;
    align-items: center;
    gap: 3px;
    font-size: 11px;
    color: #38bdf8;
    background: rgba(56, 189, 248, 0.1);
    border: 1px solid rgba(56, 189, 248, 0.25);
    padding: 3px 8px;
    border-radius: 4px;
    text-decoration: none;
    transition: all 0.15s;
    font-weight: 600;
    white-space: nowrap;
    flex-shrink: 0;
}}


.bridge-badge {{
    display: inline-block;
    font-size: 8.5px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.2px;
    color: #94a3b8;
    background: rgba(148, 163, 184, 0.12);
    border: 1px dashed rgba(148, 163, 184, 0.4);
    border-radius: 3px;
    padding: 0 4px;
    margin-left: 4px;
    vertical-align: middle;
    line-height: 1.4;
    flex-shrink: 0;
}}
.non-ciliary-badge {{
    display: inline-block;
    font-size: 8.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    color: #f59e0b;
    background: rgba(245, 158, 11, 0.15);
    border: 1px dashed rgba(245, 158, 11, 0.4);
    border-radius: 3px;
    padding: 0 4px;
    margin-left: 4px;
    vertical-align: middle;
    line-height: 1.4;
    flex-shrink: 0;
}}

.cilia-badge {{
    display: inline-block;
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    color: #34d399;
    background: rgba(52, 211, 153, 0.14);
    border: 1px solid rgba(52, 211, 153, 0.35);
    border-radius: 3px;
    padding: 0 4px;
    margin-left: 4px;
    vertical-align: middle;
    flex-shrink: 0;
    line-height: 1.4;
}}

.gene-ext-link:hover {{
    background: rgba(56, 189, 248, 0.2);
    border-color: #38bdf8;
    color: #ffffff;
    text-decoration: none;
}}

.btn-info-circle {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background: rgba(56, 189, 248, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.38);
    color: #38bdf8;
    font-size: 13px;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.15s ease;
    padding: 0;
    margin-left: 4px;
    vertical-align: middle;
    line-height: 1;
    flex-shrink: 0;
}}
.btn-info-circle:hover {{
    background: rgba(56, 189, 248, 0.3);
    border-color: #38bdf8;
    color: #ffffff;
    box-shadow: 0 0 10px rgba(56, 189, 248, 0.45);
    transform: scale(1.08);
}}
.modal-overlay {{
    display: none;
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(3, 7, 18, 0.82);
    backdrop-filter: blur(5px);
    z-index: 9999;
    align-items: center;
    justify-content: center;
    padding: 20px;
}}
.modal-overlay.active {{
    display: flex;
}}
.modal-container {{
    background: #111827;
    border: 1px solid #2a3558;
    border-radius: 12px;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.8), 0 0 24px rgba(56, 189, 248, 0.15);
    width: 100%;
    max-width: 690px;
    max-height: 92vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    animation: modalScaleIn 0.18s cubic-bezier(0.16, 1, 0.3, 1);
}}
@keyframes modalScaleIn {{
    from {{ transform: scale(0.95); opacity: 0; }}
    to {{ transform: scale(1); opacity: 1; }}
}}
.modal-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 18px;
    border-bottom: 1px solid #1f293d;
    background: #0d1322;
}}
.modal-close {{
    background: transparent;
    border: none;
    color: #94a3b8;
    font-size: 24px;
    line-height: 1;
    cursor: pointer;
    padding: 2px 6px;
    border-radius: 4px;
    transition: all 0.12s;
}}
.modal-close:hover {{
    color: #ffffff;
    background: rgba(255, 255, 255, 0.1);
}}
.modal-body {{
    padding: 16px 18px;
    overflow-y: auto;
}}
.dataset-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 11.5px;
    color: #cbd5e1;
    margin-top: 10px;
}}
.dataset-table th {{
    background: #1a2238;
    color: #94a3b8;
    text-align: left;
    padding: 7px 10px;
    font-weight: 600;
    border-bottom: 1px solid #2a3558;
}}
.dataset-table td {{
    padding: 7px 10px;
    border-bottom: 1px solid #1a2238;
    vertical-align: middle;
}}
.dataset-table tr:hover td {{
    background: rgba(30, 41, 59, 0.45);
}}
.badge-sub {{
    display: inline-block;
    font-size: 9px;
    padding: 1px 4px;
    border-radius: 3px;
    background: rgba(255, 255, 255, 0.08);
    color: #94a3b8;
    margin-left: 3px;
}}
.btn-sm {{
    padding: 2px 8px;
    font-size: 11px;
}}
#status-bar {{
    background: #111827;
    border-top: 1px solid #2a3050;
    padding: 5px 15px;
    font-size: 11px;
    color: #556;
    display: flex;
    gap: 20px;
}}
.empty-state {{
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: #3a4570;
    font-size: 14px;
    text-align: center;
    padding: 20px;
}}
/* Custom scrollbar */
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: #2a3050; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: #3a4570; }}
/* Tooltip */
.cy-tooltip {{
    position: absolute;
    background: #1a2040;
    border: 1px solid #3a5080;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 12px;
    color: #e0e6f0;
    pointer-events: none;
    z-index: 50;
    white-space: nowrap;
    display: none;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
}}
/* Toast notification */
#toast {{
    position: fixed;
    top: 60px;
    left: 50%;
    transform: translateX(-50%);
    background: #2a1a00;
    border: 1px solid #6a5a00;
    color: #ffd866;
    padding: 10px 20px;
    border-radius: 8px;
    font-size: 13px;
    z-index: 200;
    opacity: 0;
    transition: opacity 0.3s;
    pointer-events: none;
    text-align: center;
    max-width: 500px;
}}
#toast.show {{
    opacity: 1;
}}
/* Legend */
#legend {{
    position: absolute;
    bottom: 16px;
    left: 16px;
    background: rgba(17, 24, 39, 0.92);
    backdrop-filter: blur(8px);
    border: 1px solid #2a3558;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 11px;
    color: #cbd5e1;
    z-index: 15;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
    width: 230px;
    user-select: none;
}}
.legend-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-weight: 600;
    color: #93c5fd;
    font-size: 11.5px;
}}
.legend-section {{
    margin-top: 8px;
}}
.legend-title {{
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #8892b0;
    margin-bottom: 4px;
}}
.legend-bar {{
    height: 6px;
    border-radius: 3px;
    margin-bottom: 3px;
}}
.legend-labels {{
    display: flex;
    justify-content: space-between;
    font-size: 9.5px;
    color: #64748b;
}}
.legend-items {{
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin-top: 6px;
    padding-top: 6px;
    border-top: 1px solid #1e293b;
}}
.legend-item {{
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: 10.5px;
    color: #94a3b8;
}}
.legend-dot {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}}
.legend-line {{
    width: 16px;
    height: 2.5px;
    border-radius: 1px;
    flex-shrink: 0;
}}
</style>
</head>
<body>
<div id="header">
    <h1>🧬 Gene Loss Network</h1>
    <button class="btn btn-clusters-header" onclick="showAllClusters()" id="btn-all-clusters-head" title="Show all Leiden clusters at once">🔬 All Clusters (Leiden)</button>
    <div class="search-box">
        <input type="text" id="search" placeholder="Search gene (e.g. SCAPER, CEP290)..." autocomplete="off">
        <div id="suggestions"></div>
    </div>
    <div class="controls">
        <label>Min Jaccard: <span id="thresh-val">0.10</span></label>
        <input type="range" id="thresh" min="0.05" max="1.0" step="0.01" value="0.10">
        <label>Show Top: <span id="topn-val">25</span></label>
        <input type="range" id="topn" min="1" max="{top_k}" step="1" value="25">
        <label style="display:inline-flex;align-items:center;gap:4px;font-size:12px;cursor:pointer;color:#cbd5e1;" title="Ignore the Show Top cap and use every stored partner above Min Jaccard (up to {top_k} per gene)">
            <input type="checkbox" id="toggle-topn-max"> Max
        </label>
        <label style="margin-left:5px;">Layout:</label>
        <select id="layout-select" class="btn" style="padding:4px 8px;">
            <option value="cose" selected>Force (Spread)</option>
            <option value="concentric">Concentric (Radial)</option>
            <option value="circle">Circle</option>
        </select>
        <label style="display:inline-flex;align-items:center;gap:4px;font-size:12px;cursor:pointer;color:#cbd5e1;margin-left:4px;">
            <input type="checkbox" id="toggle-labels" checked> Labels
        </label>
        <label style="margin-left:4px;">Genes:</label>
        <select id="gene-filter" class="btn" style="padding:4px 8px;" title="Filter cluster views by curated ciliary gene set">
            <option value="all" selected>All genes</option>
            <option value="syscilia_v2">SYSCILIA v2 (SCGSv2 - 509 genes)</option>
            <option value="syscilia_v1">SYSCILIA v1 (SCGSv1 - 275 genes)</option>
            <option value="ciliacarta">CiliaCarta (935 genes)</option>
            <option value="both">All Ciliary (1,132 genes)</option>
        </select>
        <button id="btn-cilia-info" class="btn-info-circle" title="Explain ciliary datasets & view Venn diagram" onclick="openCiliaModal()">ⓘ</button>
        <button class="btn btn-accent" onclick="fitGraph()">Fit view</button>
        <button class="btn" onclick="exportCytoscape()">Export JSON</button>
        <button class="btn btn-danger" onclick="clearGraph()">Clear graph</button>
        <button class="btn btn-clusters" onclick="showAllClusters()" id="btn-all-clusters" title="Show all Leiden clusters at once">🔬 All Clusters</button>
    </div>
</div>
<div id="main">
    <div id="cy">
        <div class="empty-state" id="empty-msg">
            <div style="max-width: 500px; padding: 24px; background: rgba(17, 24, 39, 0.85); border: 1px solid #2a3558; border-radius: 12px; box-shadow: 0 8px 32px rgba(0,0,0,0.5);">
                <div style="font-size: 36px; margin-bottom: 12px;">🧬</div>
                <div style="font-size: 18px; font-weight: 600; color: #7eb8ff; margin-bottom: 8px;">Explore Co-Loss Networks & Modules</div>
                <div style="font-size: 13px; color: #94a3b8; line-height: 1.5; margin-bottom: 20px;">
                    Search for any gene above (e.g. <strong>SCAPER</strong>, <strong>CEP290</strong>) to explore its direct co-loss network, or view the global partition across all 80 Leiden clusters.
                </div>
                <button class="btn btn-clusters-header" onclick="showAllClusters()" style="font-size: 14px; padding: 10px 24px;">
                    🔬 View All 80 Clusters (Leiden Modules)
                </button>
            </div>
        </div>
        <div id="legend">
            <div class="legend-header">
                <span>Network Legend</span>
                <button id="legend-toggle" title="Collapse / Expand" style="background:none;border:none;color:#94a3b8;cursor:pointer;font-size:14px;line-height:1;padding:0 2px;">−</button>
            </div>
            <div id="legend-body">
                <div class="legend-section">
                    <div class="legend-title">Nodes: Independent Losses</div>
                    <div class="legend-bar" style="background: linear-gradient(to right, hsl(200, 70%, 45%), hsl(145, 70%, 50%), hsl(90, 70%, 55%), hsl(40, 70%, 60%));"></div>
                    <div class="legend-labels">
                        <span>5L (rare, small)</span>
                        <span>30L</span>
                        <span>60+L (frequent)</span>
                    </div>
                </div>
                <div class="legend-section">
                    <div class="legend-title">Edges: Jaccard Similarity</div>
                    <div class="legend-bar" style="background: linear-gradient(to right, rgb(51, 255, 48), rgb(255, 255, 0), rgb(255, 0, 0));"></div>
                    <div class="legend-labels">
                        <span>0.10 (thin)</span>
                        <span>0.50</span>
                        <span>1.00 (thick)</span>
                    </div>
                </div>
                <div class="legend-items">
                    <div class="legend-item">
                        <span class="legend-dot" style="background:#0284c7;border:2px solid #38bdf8;"></span>
                        <span>Focus / Selected Gene</span>
                    </div>
                    <div class="legend-item">
                        <span class="legend-line" style="background:#38bdf8;"></span>
                        <span>Selected Node Edges</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div id="sidebar">
        <div id="gene-info">
            <div class="empty-state" style="height:auto;min-height:50px;font-size:12px;">
                Select a gene to see its top partners
            </div>
        </div>
        <div id="partner-list"></div>
    </div>
</div>
<div id="status-bar">
    <span id="status-nodes">Nodes: 0</span>
    <span id="status-edges">Edges: 0</span>
    <span id="status-selected">Selected: none</span>
</div>
<div class="cy-tooltip" id="tooltip"></div>
<div id="toast"></div>

<div id="cilia-modal-overlay" class="modal-overlay" onclick="closeCiliaModal(event)">
    <div class="modal-container" onclick="event.stopPropagation()">
        <div class="modal-header">
            <div style="display:flex; align-items:center; gap:8px;">
                <span style="font-size:20px;">🧬</span>
                <h3 style="margin:0; font-size:16px; color:#f1f5f9;">Ciliary Gene Datasets &amp; Intersections</h3>
            </div>
            <button class="modal-close" onclick="closeCiliaModal()" title="Close (Esc)">×</button>
        </div>
        <div class="modal-body">
            <!-- Venn Diagram SVG -->
            <div style="text-align:center;">
                <svg viewBox="0 0 650 350" width="100%" height="310" xmlns="http://www.w3.org/2000/svg" style="background:#0b1120; border-radius:10px; border:1px solid #1e293b; user-select:none;">
                  <defs>
                    <!-- Radial Gradients for Sets -->
                    <radialGradient id="grad-v2" cx="35%" cy="40%" r="65%">
                      <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.32"/>
                      <stop offset="100%" stop-color="#0284c7" stop-opacity="0.06"/>
                    </radialGradient>
                    <radialGradient id="grad-cc" cx="65%" cy="40%" r="65%">
                      <stop offset="0%" stop-color="#f59e0b" stop-opacity="0.30"/>
                      <stop offset="100%" stop-color="#d97706" stop-opacity="0.06"/>
                    </radialGradient>
                    <radialGradient id="grad-v1" cx="45%" cy="50%" r="60%">
                      <stop offset="0%" stop-color="#10b981" stop-opacity="0.40"/>
                      <stop offset="100%" stop-color="#059669" stop-opacity="0.10"/>
                    </radialGradient>
                    
                    <!-- Filter for glow / shadow -->
                    <filter id="badge-shadow" x="-20%" y="-20%" width="140%" height="140%">
                      <feDropShadow dx="0" dy="2" stdDeviation="4" flood-color="#000000" flood-opacity="0.75"/>
                    </filter>
                  </defs>

                  <!-- ================= CIRCLES ================= -->
                  <!-- CIRCLE 3: CiliaCarta (935 genes) -->
                  <circle cx="408" cy="175" r="128" fill="url(#grad-cc)" stroke="#f59e0b" stroke-width="2.2" stroke-opacity="0.9">
                    <title>CiliaCarta (935 genes): Genome-wide Bayesian integration of co-expression, genomics, and proteomics</title>
                  </circle>

                  <!-- CIRCLE 1: SYSCILIA v2 (509 genes) -->
                  <circle cx="225" cy="175" r="112" fill="url(#grad-v2)" stroke="#38bdf8" stroke-width="2.2" stroke-opacity="0.9">
                    <title>SYSCILIA v2 (509 genes): 2021 Gold Standard (408 First-order + 102 Second-order)</title>
                  </circle>

                  <!-- CIRCLE 2: SYSCILIA v1 (275 genes) - sits in overlap -->
                  <circle cx="280" cy="198" r="68" fill="url(#grad-v1)" stroke="#10b981" stroke-width="2.2" stroke-opacity="0.95">
                    <title>SYSCILIA v1 (275 genes): Original 2013 Gold Standard (227 verified + 48 candidates)</title>
                  </circle>

                  <!-- ================= SET TITLES ================= -->
                  <!-- SCGSv2 Title (Top Left) -->
                  <g transform="translate(150, 16)" text-anchor="middle">
                    <text fill="#38bdf8" font-size="13" font-weight="800" letter-spacing="0.3">SYSCILIA v2</text>
                    <text y="13" fill="#7dd3fc" font-size="10.5" font-weight="600">(SCGSv2: 509 genes)</text>
                  </g>

                  <!-- CiliaCarta Title (Top Right) -->
                  <g transform="translate(485, 16)" text-anchor="middle">
                    <text fill="#f59e0b" font-size="13.5" font-weight="800" letter-spacing="0.3">CiliaCarta</text>
                    <text y="13" fill="#fcd34d" font-size="10.5" font-weight="600">(CiliaCarta.csv: 935 genes)</text>
                  </g>

                  <!-- SCGSv1 Title & Pointer Line (Bottom) -->
                  <g transform="translate(235, 314)" text-anchor="middle">
                    <text fill="#34d399" font-size="12.5" font-weight="800">SYSCILIA v1</text>
                    <text y="13" fill="#6ee7b7" font-size="9.5" font-weight="600">(SCGSv1: 275 genes)</text>
                  </g>
                  <polyline points="255,302 272,278 278,258" stroke="#10b981" stroke-width="1.3" stroke-dasharray="3 2" fill="none"/>
                  <circle cx="278" cy="258" r="2.5" fill="#10b981"/>

                  <!-- ================= CALLOUT 1: 85 genes (SCGSv2 & CiliaCarta only) ================= -->
                  <!-- Placed at top with a crisp elbow leader line pointing into the upper overlap lens -->
                  <g transform="translate(315, 34)" text-anchor="middle">
                    <rect x="-46" y="-13" width="92" height="28" rx="5" fill="#0b1120" fill-opacity="0.95" stroke="#f59e0b" stroke-width="1.2" stroke-opacity="0.9" filter="url(#badge-shadow)"/>
                    <text y="0" fill="#fde68a" font-size="12.5" font-weight="800">85 genes</text>
                    <text y="10" fill="#fcd34d" font-size="7.5" font-weight="600">v2 ∩ CiliaCarta only</text>
                    <title>v2 ∩ CiliaCarta only (85 genes): Shared by SYSCILIA v2 and CiliaCarta, not in v1</title>
                  </g>
                  <polyline points="315,49 315,75 328,100" stroke="#f59e0b" stroke-width="1.3" fill="none"/>
                  <circle cx="328" cy="100" r="2.5" fill="#f59e0b"/>

                  <!-- ================= CALLOUT 2: 4 genes (v1 historical candidates) ================= -->
                  <!-- Leader line from bottom left into the lower rim of v1 -->
                  <g transform="translate(95, 260)" text-anchor="start">
                    <rect x="-6" y="-13" width="86" height="28" rx="5" fill="#0b1120" fill-opacity="0.95" stroke="#34d399" stroke-width="1.2" stroke-opacity="0.8" filter="url(#badge-shadow)"/>
                    <text x="37" y="0" fill="#6ee7b7" font-size="11.5" font-weight="800" text-anchor="middle">4 genes</text>
                    <text x="37" y="10" fill="#a7f3d0" font-size="7.5" font-weight="600" text-anchor="middle">v1 candidates only</text>
                    <title>SCGSv1 candidates only (4 genes): 2013 paper predictions not retained in v2</title>
                  </g>
                  <polyline points="175,254 212,246 226,238" stroke="#34d399" stroke-width="1.3" fill="none"/>
                  <circle cx="226" cy="238" r="2.5" fill="#34d399"/>

                  <!-- ================= DIRECT REGIONS ================= -->

                  <!-- 1. TRIPLE INTERSECTION: Core Shared (238 genes) -->
                  <g transform="translate(318, 190)" text-anchor="middle">
                    <rect x="-42" y="-22" width="84" height="44" rx="7" fill="#070d1e" fill-opacity="0.96" stroke="#38bdf8" stroke-width="1.4" filter="url(#badge-shadow)"/>
                    <text y="-4" fill="#ffffff" font-size="16" font-weight="900" letter-spacing="0.3">238</text>
                    <text y="9" fill="#93c5fd" font-size="9" font-weight="700">Core Shared</text>
                    <text y="18" fill="#7dd3fc" font-size="7.5" opacity="0.85">v1 ∩ v2 ∩ CC</text>
                    <title>Core Shared (238 genes): Supported by SYSCILIA v1, SYSCILIA v2, and CiliaCarta (e.g. IFT88, BBS1, CEP290)</title>
                  </g>

                  <!-- 2. CILIACARTA ONLY (611 genes) -->
                  <g transform="translate(476, 175)" text-anchor="middle">
                    <text y="-6" fill="#fde047" font-size="18" font-weight="900">611</text>
                    <text y="10" fill="#fef08a" font-size="10" font-weight="700">CiliaCarta only</text>
                    <text y="23" fill="#fcd34d" font-size="8.5" opacity="0.8">Bayesian ML &amp; GO</text>
                    <title>CiliaCarta Exclusive (611 genes): Genome-wide Bayesian predictions and GO annotations outside SYSCILIA gold standards</title>
                  </g>

                  <!-- 3. SCGSv2 ONLY (154 genes, e.g. SCAPER) -->
                  <g transform="translate(156, 172)" text-anchor="middle">
                    <text y="-6" fill="#7dd3fc" font-size="17" font-weight="900">154</text>
                    <text y="9" fill="#38bdf8" font-size="9.5" font-weight="700">SCGSv2 only</text>
                    <text y="21" fill="#bae6fd" font-size="8.5" font-style="italic">(includes SCAPER)</text>
                    <title>SCGSv2 Exclusive (154 genes): Modern ciliopathy and regulatory genes added in 2021 Gold Standard (e.g. SCAPER)</title>
                  </g>

                  <!-- 4. SCGSv1 & SCGSv2 ONLY (32 genes) -->
                  <g transform="translate(242, 198)" text-anchor="middle">
                    <rect x="-24" y="-12" width="48" height="24" rx="4" fill="#061c18" fill-opacity="0.75" stroke="#10b981" stroke-width="0.8"/>
                    <text y="2" fill="#a7f3d0" font-size="12" font-weight="800">32</text>
                    <text y="10" fill="#6ee7b7" font-size="7" font-weight="600">v1 ∩ v2</text>
                    <title>SCGSv1 ∩ SCGSv2 only (32 genes): High-confidence components in both SYSCILIA versions outside CiliaCarta</title>
                  </g>

                  <!-- ================= BOTTOM LEGEND BAR ================= -->
                  <g transform="translate(32, 336)" font-size="9.5" fill="#94a3b8">
                    <circle cx="0" cy="0" r="4" fill="#38bdf8"/>
                    <text x="8" y="3.5">SCGSv2: <tspan fill="#e2e8f0" font-weight="700">509</tspan></text>
                    
                    <circle cx="105" cy="0" r="4" fill="#10b981"/>
                    <text x="113" y="3.5">SCGSv1: <tspan fill="#e2e8f0" font-weight="700">275</tspan></text>
                    
                    <circle cx="210" cy="0" r="4" fill="#f59e0b"/>
                    <text x="218" y="3.5">CiliaCarta: <tspan fill="#e2e8f0" font-weight="700">935</tspan></text>
                    
                    <circle cx="342" cy="0" r="4" fill="#a855f7"/>
                    <text x="350" y="3.5">All Ciliary Union: <tspan fill="#38bdf8" font-weight="700">1,132</tspan> genes</text>
                    <text x="520" y="3.5" fill="#64748b" font-size="8.5">(+7 curated in CSV)</text>
                  </g>
                </svg>
            </div>

            <!-- Explanatory Breakdown Table -->
            <table class="dataset-table">
                <thead>
                    <tr>
                        <th>Option</th>
                        <th>Genes</th>
                        <th>In Matrix (L &ge; 5)</th>
                        <th>Description &amp; Key Evidence</th>
                        <th>Quick Select</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong style="color:#7dd3fc;">SYSCILIA v2</strong><br><span class="badge-sub">SCGSv2</span></td>
                        <td><strong>509</strong></td>
                        <td>310</td>
                        <td>2021 Gold Standard: <strong>408</strong> First-order (core/structural) + <strong>102</strong> Second-order (regulatory). Includes recent ciliopathy genes (e.g. <strong>SCAPER</strong>).</td>
                        <td><button class="btn btn-sm btn-accent" onclick="selectFilterAndClose('syscilia_v2')">Select</button></td>
                    </tr>
                    <tr>
                        <td><strong style="color:#6ee7b7;">SYSCILIA v1</strong><br><span class="badge-sub">SCGSv1</span></td>
                        <td><strong>275</strong></td>
                        <td>186</td>
                        <td>Original 2013 Gold Standard (van Dam et al.): 227 confirmed ciliary components + 48 high-confidence candidate predictions.</td>
                        <td><button class="btn btn-sm" onclick="selectFilterAndClose('syscilia_v1')">Select</button></td>
                    </tr>
                    <tr>
                        <td><strong style="color:#fde047;">CiliaCarta</strong><br><span class="badge-sub">CiliaCarta.csv</span></td>
                        <td><strong>935</strong></td>
                        <td>514</td>
                        <td>Complete compendium (van Dam et al. 2019): Bayesian integration of co-expression, comparative genomics, and proteomics across 935 ciliary genes.</td>
                        <td><button class="btn btn-sm" onclick="selectFilterAndClose('ciliacarta')">Select</button></td>
                    </tr>
                    <tr>
                        <td><strong style="color:#38bdf8;">All Ciliary</strong><br><span class="badge-sub">Union</span></td>
                        <td><strong>1,132</strong></td>
                        <td>626</td>
                        <td>Comprehensive union of SYSCILIA gold standards (v1 + v2, 521 genes) and CiliaCarta (935 genes).</td>
                        <td><button class="btn btn-sm" onclick="selectFilterAndClose('both')">Select</button></td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
</div>

<script>
function openCiliaModal() {{
    const el = document.getElementById('cilia-modal-overlay');
    if (el) el.classList.add('active');
}}
function closeCiliaModal(e) {{
    if (e && e.target && e.target !== document.getElementById('cilia-modal-overlay') && !e.target.classList.contains('modal-close')) {{
        return;
    }}
    const el = document.getElementById('cilia-modal-overlay');
    if (el) el.classList.remove('active');
}}
function selectFilterAndClose(val) {{
    const sel = document.getElementById('gene-filter');
    if (sel) {{
        sel.value = val;
        sel.dispatchEvent(new Event('change'));
    }}
    const el = document.getElementById('cilia-modal-overlay');
    if (el) el.classList.remove('active');
}}
document.addEventListener('keydown', function(e) {{
    if (e.key === 'Escape') {{
        const el = document.getElementById('cilia-modal-overlay');
        if (el && el.classList.contains('active')) {{
            el.classList.remove('active');
        }}
    }}
}});
// ---- Embedded data ----
const G = {data_json};
const NAMES = {names_json};
const CLUSTERS = {clusters_json};
const GENE_CL = {gene_cluster_json};
const CLUSTER_NAMES = {cluster_names_json};
const HAS_CLUSTERS = Object.keys(CLUSTERS).length > 0;
const CILIARY_SETS = {ciliary_sets_json};
const CILIA_INFO = {cilia_info_json};
const ALL_CILIARY = new Set(CILIARY_SETS["all_ciliary"] || CILIARY_SETS["both"] || []);
let lastClickedClusterId = null;
let geneFilterMode = 'all';

function getCiliaBadgeHtml(geneName) {{
    if (!ALL_CILIARY.has(geneName)) return '';
    const info = (typeof CILIA_INFO !== 'undefined' && CILIA_INFO[geneName]) ? CILIA_INFO[geneName] : null;
    let title = 'Curated ciliary component (SYSCILIA / CiliaCarta)';
    if (info) {{
        const parts = [];
        if (info.v2) parts.push(`SCGSv2 (${{info.v2}})`);
        if (info.v1) parts.push('SCGSv1');
        if (info.cc) parts.push('CiliaCarta');
        const dsText = parts.length ? parts.join(', ') : 'Curated Ciliary';
        title = `Curated ciliary component: ${{dsText}}`;
        if (info.loc) {{
            title += `\\nLocalization: ${{info.loc}}`;
        }}
    }}
    return `<span class="cilia-badge" title="${{title.replace(/"/g, '&quot;')}}">cilia</span>`;
}}

function getActiveGeneSet() {{
    if (geneFilterMode === 'all') return null;
    return new Set(CILIARY_SETS[geneFilterMode] || []);
}}

function getClusterMembers(cid) {{
    const activeSet = getActiveGeneSet();
    return (CLUSTERS[cid] || []).filter(name => G[name] && (!activeSet || activeSet.has(name)));
}}

function getVisibleClusterIds() {{
    return Object.keys(CLUSTERS).map(Number)
        .filter(cid => getClusterMembers(cid).length > 0)
        .sort((a, b) => getClusterMembers(b).length - getClusterMembers(a).length);
}}

// Pre-assign cluster colors consistently
const clusterColors = {{}};
const sortedClusterIds = Object.keys(CLUSTERS).map(Number).sort((a, b) => CLUSTERS[b].length - CLUSTERS[a].length);
sortedClusterIds.forEach((cid, i) => {{
    const h = (i * 360 / Math.max(1, sortedClusterIds.length) + 15) % 360;
    clusterColors[cid] = `hsl(${{Math.round(h)}}, 65%, 55%)`;
}});

function getGeneClusterHtml(geneName) {{
    if (!HAS_CLUSTERS) return '';
    const cid = GENE_CL[geneName];
    if (cid === undefined) return '';
    const cname = CLUSTER_NAMES[cid] || `Cluster ${{cid}}`;
    const csize = CLUSTERS[cid] ? CLUSTERS[cid].length : 0;
    const color = clusterColors[cid] || '#38bdf8';
    const escaped = geneName.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
    return `
    <div style="margin-top:9px; padding:7px 10px; background:rgba(26,32,64,0.85); border:1px solid #2a3558; border-radius:6px; display:flex; align-items:center; justify-content:space-between; gap:8px;">
        <div style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:11.5px; color:#cbd5e1;" title="Cluster ${{cid}}: ${{cname}} (${{csize}} genes)">
            <span style="display:inline-block; width:9px; height:9px; border-radius:2px; background:${{color}}; margin-right:5px; vertical-align:middle;"></span>
            <strong style="color:#7eb8ff;">C${{cid}}:</strong> ${{cname}} <span style="color:#8892b0;">(${{csize}} genes)</span>
        </div>
        <button class="btn btn-accent" style="padding:3px 9px; font-size:11px; white-space:nowrap; flex-shrink:0; font-weight:600;" data-cid="${{cid}}" data-gene="${{escaped}}" onclick="showSingleCluster(parseInt(this.dataset.cid), this.dataset.gene)">View Cluster →</button>
    </div>`;
}}

function getUniProtUrl(gene) {{
    if (!gene) return 'https://www.uniprot.org';
    const trimmed = gene.trim();
    if (/^[A-Za-z0-9_-]+$/.test(trimmed)) {{
        return `https://www.uniprot.org/uniprotkb?query=gene_exact:${{encodeURIComponent(trimmed)}}+AND+organism_id:9606`;
    }}
    return `https://www.uniprot.org/uniprotkb?query=${{encodeURIComponent(trimmed)}}+AND+organism_id:9606`;
}}

// ---- State & DOM references ----
let cy = null;
let selectedGene = null;
const graphGenes = new Set();
const graphEdges = new Set();
const expandedGenes = [];  // ordered list of genes user explicitly expanded
let clusterViewActive = false;
let currentView = 'none';  // 'gene' | 'single_cluster' | 'all_clusters' | 'none'
let currentClusterId = null;
let currentClusterHighlightGene = null;
const searchInput = document.getElementById('search');
const sugBox = document.getElementById('suggestions');

// ---- Colour helpers ----
function jaccardColor(j) {{
    // blue (low) -> yellow -> red (high)
    const r = Math.min(255, Math.floor(j < 0.5 ? j * 2 * 255 : 255));
    const g = Math.min(255, Math.floor(j < 0.5 ? 255 : (1 - j) * 2 * 255));
    const b = Math.floor(Math.max(0, (0.3 - j) * 3 * 80));
    return `rgb(${{r}},${{g}},${{b}})`;
}}

function lossColor(l) {{
    const t = Math.min(l / 60, 1);
    const h = Math.floor(200 - t * 160);  // 200 (blue) -> 40 (orange)
    return `hsl(${{h}}, 70%, ${{45 + t * 15}}%)`;
}}

function lossSize(l) {{
    return 6 + Math.sqrt(l) * 1.4;
}}

let toastTimer = null;
function showToast(msg, duration) {{
    duration = duration || 3000;
    const t = document.getElementById('toast');
    t.textContent = msg;
    t.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.classList.remove('show'), duration);
}}

// ---- Cytoscape init ----
function initCy() {{
    cy = cytoscape({{
        container: document.getElementById('cy'),
        style: [
            {{
                selector: 'node',
                style: {{
                    'label': 'data(label)',
                    'width': 'data(size)',
                    'height': 'data(size)',
                    'background-color': 'data(color)',
                    'border-width': 1,
                    'border-color': '#ffffff',
                    'border-opacity': 0.35,
                    'font-size': 10,
                    'color': '#e2e8f0',
                    'text-valign': 'bottom',
                    'text-margin-y': 4,
                    'text-outline-width': 2,
                    'text-outline-color': '#0a0e17',
                    'min-zoomed-font-size': 6,
                    'text-max-width': '90px',
                    'text-wrap': 'ellipsis',
                    'text-opacity': 1,
                }}
            }},
            {{
                selector: 'node.hide-label',
                style: {{
                    'text-opacity': 0,
                }}
            }},
            {{
                selector: 'node.focus',
                style: {{
                    'border-width': 3,
                    'border-color': '#38bdf8',
                    'border-opacity': 1,
                    'font-size': 13,
                    'font-weight': 'bold',
                    'color': '#38bdf8',
                    'text-opacity': 1,
                    'z-index': 25,
                    'min-zoomed-font-size': 0,
                }}
            }},
            {{
                selector: 'node.hopper',
                style: {{
                    'opacity': 0.38,
                    'background-color': '#475569',
                    'border-width': 1.5,
                    'border-style': 'dashed',
                    'border-color': '#94a3b8',
                    'color': '#94a3b8',
                    'text-opacity': 0.75,
                }}
            }},
            {{
                selector: 'edge.hopper-edge',
                style: {{
                    'line-style': 'dashed',
                    'opacity': 0.20,
                    'line-color': '#94a3b8',
                }}
            }},
            {{
                selector: 'node:selected',
                style: {{
                    'border-width': 3,
                    'border-color': '#38bdf8',
                    'border-opacity': 1,
                    'font-size': 12,
                    'font-weight': 'bold',
                    'color': '#ffffff',
                    'text-opacity': 1,
                    'z-index': 20,
                    'min-zoomed-font-size': 0,
                }}
            }},
            {{
                selector: 'node:active',
                style: {{
                    'overlay-opacity': 0,
                }}
            }},
            {{
                selector: 'edge',
                style: {{
                    'width': 'data(width)',
                    'line-color': 'data(color)',
                    'opacity': 0.25,
                    'curve-style': 'haystack',
                    'haystack-radius': 0.5,
                }}
            }},
            {{
                selector: 'edge:selected',
                style: {{
                    'width': 2.5,
                    'opacity': 0.9,
                    'z-index': 15,
                }}
            }},
            {{
                selector: 'node.cluster-label',
                style: {{
                    'background-opacity': 0,
                    'border-width': 0,
                    'font-size': 11,
                    'color': '#aab',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'text-wrap': 'wrap',
                    'text-max-width': '150px',
                    'text-opacity': 0.9,
                    'width': 1,
                    'height': 1,
                    'min-zoomed-font-size': 4,
                }}
            }}
        ],
        layout: {{ name: 'preset' }},
        minZoom: 0.1,
        maxZoom: 5,
        wheelSensitivity: 0.3,
    }});

    cy.on('tap', 'node', function(evt) {{
        const d = evt.target.data();
        if (d.isLabel) {{
            handleClusterClick(d.clusterId);
            return;
        }}
        if (currentView === 'all_clusters' && d.clusterId !== undefined) {{
            handleClusterClick(d.clusterId, d.id);
            return;
        }}
        const name = d.id;
        if (name && G[name]) renderNetwork(name);
    }});

    // Tooltip on hover
    const tooltip = document.getElementById('tooltip');
    cy.on('mouseover', 'node', function(evt) {{
        const d = evt.target.data();
        if (d.isLabel) {{
            const isZoomed = lastClickedClusterId === d.clusterId;
            tooltip.innerHTML = `<strong>${{d.fullName}}</strong><br>${{isZoomed ? 'Click to open full cluster' : 'Click to zoom in (click again for full cluster)'}}`;
            tooltip.style.display = 'block';
            return;
        }}
        const cid = d.id ? GENE_CL[d.id] : undefined;
        const clName = (cid !== undefined && CLUSTER_NAMES[cid]) ? `: ${{CLUSTER_NAMES[cid]}}` : '';
        const clInfo = (cid !== undefined) ? `<br><span style="color:#7eb8ff;">Cluster ${{cid}}</span>${{clName}}` : '';
        const nonCilTag = d.isNonCiliary ? '<br><span style="color:#f59e0b;font-weight:600;">⚠️ Non-ciliary bridge (connects ciliary genes)</span>' : '';
        tooltip.innerHTML = `<strong>${{d.label}}</strong>${{nonCilTag}}<br>Losses: ${{d.losses}}${{clInfo}}`;
        tooltip.style.display = 'block';
    }});
    cy.on('mouseover', 'edge', function(evt) {{
        const d = evt.target.data();
        tooltip.innerHTML = `${{d.source_name}} ↔ ${{d.target_name}}<br>Jaccard: ${{d.jaccard.toFixed(4)}}`;
        tooltip.style.display = 'block';
    }});
    cy.on('mousemove', function(evt) {{
        tooltip.style.left = (evt.originalEvent.clientX + 12) + 'px';
        tooltip.style.top = (evt.originalEvent.clientY + 12) + 'px';
    }});
    cy.on('mouseout', function() {{
        tooltip.style.display = 'none';
    }});

    document.getElementById('empty-msg').style.display = 'flex';
}}

// ---- Add gene node to graph ----
function addGeneNode(name, isFocus, isHopper) {{
    if (!G[name] || graphGenes.has(name)) return;
    const info = G[name];
    graphGenes.add(name);

    const shortLabel = name.length > 14 ? name.slice(0, 12) + '…' : name;
    const showLabels = document.getElementById('toggle-labels') ? document.getElementById('toggle-labels').checked : true;

    let classes = [];
    if (isFocus) classes.push('focus');
    if (isHopper) classes.push('hopper');
    if (!showLabels && !isFocus) classes.push('hide-label');

    cy.add({{
        group: 'nodes',
        data: {{
            id: name,
            label: isFocus ? name : shortLabel,
            fullName: name,
            losses: info.l,
            size: lossSize(info.l) * (isFocus ? 1.35 : 1.0),
            color: isHopper ? '#475569' : lossColor(info.l),
            isHopper: !!isHopper,
        }},
        classes: classes.join(' ')
    }});
}}

function addEdge(a, b, j, isHopperEdge) {{
    const eid = a < b ? `${{a}}||${{b}}` : `${{b}}||${{a}}`;
    if (graphEdges.has(eid)) return;
    graphEdges.add(eid);

    cy.add({{
        group: 'edges',
        data: {{
            id: eid,
            source: a < b ? a : b,
            target: a < b ? b : a,
            jaccard: j,
            source_name: a < b ? a : b,
            target_name: a < b ? b : a,
            width: isHopperEdge ? 0.8 : (0.5 + j * 2.5),
            color: isHopperEdge ? '#94a3b8' : jaccardColor(j),
        }},
        classes: isHopperEdge ? 'hopper-edge' : ''
    }});
}}

// ---- Render Network ----
function renderNetwork(geneName) {{
    if (!geneName || !G[geneName]) return;
    currentView = 'gene';
    clusterViewActive = false;
    currentClusterId = null;
    currentClusterHighlightGene = null;
    selectedGene = geneName;
    searchInput.value = geneName;

    renderEgoNetwork(geneName);
}}

// ---- Show Top: returns Infinity when "Max" is checked, else the slider value ----
function getTopN() {{
    if (document.getElementById('toggle-topn-max').checked) return Infinity;
    return parseInt(document.getElementById('topn').value);
}}


// ---- Ego view: strictly the Focus Gene + Current Sidebar Partners ----
function renderEgoNetwork(geneName) {{
    const thresh = parseFloat(document.getElementById('thresh').value);
    const topn = getTopN();
    const info = G[geneName];

    // Filter partners strictly above threshold, and matching active ciliary set
    const activeSet = getActiveGeneSet();
    const qualifying = info.p.filter(p => p.j >= thresh && (!activeSet || activeSet.has(p.n)));
    const filtered = qualifying.slice(0, topn);
    const totalAbove = qualifying.length;

    // 1. Render Sidebar
    const dispGene = geneName.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    const isFocusCil = ALL_CILIARY.has(geneName);
    const filterNotice = geneFilterMode !== 'all' && document.getElementById('gene-filter') ? ` (filtered by ${{document.getElementById('gene-filter').selectedOptions[0].text}})` : '';
    document.getElementById('gene-info').innerHTML = `
        <div style="display:flex; align-items:center; justify-content:space-between; gap:8px;">
            <h2 style="margin-bottom:0; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; display:flex; align-items:center; gap:6px;">
                <span style="overflow:hidden; text-overflow:ellipsis;">${{dispGene}}</span>
                ${{getCiliaBadgeHtml(geneName)}}
            </h2>
            <a href="${{getUniProtUrl(geneName)}}" target="_blank" rel="noopener noreferrer" class="gene-ext-link" title="Open ${{dispGene}} on UniProt">UniProt ↗</a>
        </div>
        <div class="meta" style="margin-top:4px;">
            Independent losses: <strong>${{info.l}}</strong> &nbsp;|&nbsp;
            Partners shown: <strong>${{filtered.length}}</strong> of ${{totalAbove}} above threshold${{filterNotice}}
            ${{totalAbove > topn ? ' (increase "Show Top" to see more)' : ''}}
        </div>
        ${{getGeneClusterHtml(geneName)}}
    `;

    let html = '';
    filtered.forEach((p, i) => {{
        const pct = (p.j * 100).toFixed(0);
        const escaped = p.n.replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
        const dispName = p.n.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
        html += `
        <div class="partner" data-gene="${{escaped}}">
            <span class="rank">#${{i + 1}}</span>
            <span class="pname in-graph">${{dispName}}</span>
            ${{getCiliaBadgeHtml(p.n)}}
            <div class="jaccard-bar">
                <div class="fill" style="width:${{pct}}%; background:${{jaccardColor(p.j)}};"></div>
            </div>
            <span class="jaccard-val">${{p.j.toFixed(3)}}</span>
            <span class="ploss">${{p.l}}L</span>
            <a href="${{getUniProtUrl(p.n)}}" target="_blank" rel="noopener noreferrer" class="partner-ext-link" title="Open ${{dispName}} on UniProt" onclick="event.stopPropagation();">↗</a>
        </div>`;
    }});
    document.getElementById('partner-list').innerHTML = html;

    // 2. Render Graph - focus gene + direct partners (direct 1-hop partners are never dimmed!)
    cy.elements().remove();
    graphGenes.clear();
    graphEdges.clear();
    document.getElementById('empty-msg').style.display = 'none';

    // Focus gene and direct partners are never dimmed
    addGeneNode(geneName, true, false);

    for (const p of filtered) {{
        addGeneNode(p.n, false, false);
        addEdge(geneName, p.n, p.j, false);
    }}

    // Cross-link: add edges between any two partner nodes in the list if j >= thresh
    for (const p of filtered) {{
        const pinfo = G[p.n];
        if (!pinfo) continue;
        for (const p2 of pinfo.p) {{
            if (p2.j >= thresh && graphGenes.has(p2.n)) {{
                addEdge(p.n, p2.n, p2.j);
            }}
        }}
    }}

    // Highlight focus gene in Cytoscape
    const centerNode = cy.getElementById(geneName);
    if (centerNode.length) {{
        centerNode.select();
        centerNode.connectedEdges().select();
    }}

    // Toast notification if no partners qualify
    if (filtered.length === 0) {{
        const best = info.p.length > 0 ? info.p[0] : null;
        if (best) {{
            showToast(`No partners for ${{geneName}} above Jaccard ≥ ${{thresh.toFixed(2)}}. Best partner: ${{best.n}} at ${{best.j.toFixed(3)}}. Try lowering the threshold.`, 4500);
        }} else {{
            showToast(`${{geneName}} has no co-loss partners in the dataset.`, 3000);
        }}
    }}

    runLayout();
    updateStatus();
}}

// ---- Layout ----
function runLayout(randomize) {{
    if (randomize === undefined) randomize = true;
    if (!cy || cy.nodes().length === 0) return;

    const layoutMode = document.getElementById('layout-select') ? document.getElementById('layout-select').value : 'cose';

    let layoutConfig = {{}};
    if (layoutMode === 'concentric') {{
        layoutConfig = {{
            name: 'concentric',
            animate: true,
            animationDuration: 500,
            fit: randomize,
            padding: 60,
            concentric: function(node) {{
                if (node.data('id') === selectedGene) return 100;
                return Math.round((node.data('losses') || 0) / 5) + 1;
            }},
            levelWidth: function() {{ return 2; }},
            minNodeSpacing: 35,
        }};
    }} else if (layoutMode === 'circle') {{
        layoutConfig = {{
            name: 'circle',
            animate: true,
            animationDuration: 500,
            fit: randomize,
            padding: 60,
            spacingFactor: 1.2,
        }};
    }} else {{
        // Force-directed (cose)
        layoutConfig = {{
            name: 'cose',
            animate: true,
            animationDuration: 500,
            randomize: randomize,
            componentSpacing: 100,
            nodeRepulsion: function(node) {{ return 800000; }},
            nodeOverlap: 40,
            idealEdgeLength: function(edge) {{ return 160; }},
            edgeElasticity: function(edge) {{ return 20; }},
            nestingFactor: 1.2,
            gravity: 0.1,
            numIter: randomize ? 1000 : 500,
            initialTemp: randomize ? 1000 : 200,
            coolingFactor: 0.95,
            minTemp: 1.0,
            fit: randomize,
            padding: 60,
        }};
    }}

    const layout = cy.layout(layoutConfig);
    if (randomize) {{
        layout.promiseOn('layoutstop').then(() => {{
            cy.animate({{
                fit: {{ padding: 60 }}
            }}, {{ duration: 300 }});
        }});
    }}
    layout.run();
}}

function fitGraph() {{
    if (cy && cy.nodes().length > 0) cy.fit(undefined, 50);
}}

// ---- Clear ----
function clearGraph() {{
    cy.elements().remove();
    graphGenes.clear();
    graphEdges.clear();
    selectedGene = null;
    currentView = 'none';
    currentClusterId = null;
    currentClusterHighlightGene = null;
    clusterViewActive = false;
    searchInput.value = '';
    document.getElementById('gene-info').innerHTML =
        '<div class="empty-state" style="height:auto;min-height:50px;font-size:12px;">Select a gene to see its top partners</div>';
    document.getElementById('partner-list').innerHTML = '';
    document.getElementById('empty-msg').style.display = 'flex';
    updateStatus();
}}

// ---- Export ----
function exportCytoscape() {{
    const elements = cy.json().elements;
    const blob = new Blob([JSON.stringify(elements, null, 2)], {{ type: 'application/json' }});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'network_export.cyjs';
    a.click();
    URL.revokeObjectURL(a.href);
}}

// ---- All Clusters View ----
function clusterHue(idx, total) {{
    // Evenly spaced hues with a pleasant saturation
    return (idx * 360 / total + 15) % 360;
}}

function showAllClusters(isFilterUpdate) {{
    if (!HAS_CLUSTERS) {{
        showToast('No cluster data available. Run leiden_cluster.py first.', 3000);
        return;
    }}
    currentView = 'all_clusters';
    clusterViewActive = true;
    currentClusterId = null;
    currentClusterHighlightGene = null;
    selectedGene = null;
    searchInput.value = '';
    if (!isFilterUpdate) lastClickedClusterId = null;

    const thresh = parseFloat(document.getElementById('thresh').value);
    const MAX_NODES_PER_CLUSTER = 30;
    const MAX_EDGES_PER_CLUSTER = 25;

    // Sort clusters by size descending
    const clusterIds = getVisibleClusterIds();
    const nClusters = clusterIds.length;

    // In-place edge adjustment if already rendered
    if (isFilterUpdate && cy && cy.nodes().length > 0) {{
        // Remove edges below new threshold
        cy.edges().forEach(edge => {{
            const d = edge.data();
            if (d.jaccard < thresh) {{
                graphEdges.delete(d.id);
                cy.remove(edge);
            }}
        }});

        // Add qualifying edges up to limit
        clusterIds.forEach(cid => {{
            const members = getClusterMembers(cid);
            const color = clusterColors[cid] || '#38bdf8';
            const sortedMembers = members
                .filter(n => G[n])
                .map(n => ({{ n, l: G[n].l }}))
                .sort((a, b) => b.l - a.l);

            const visibleMembers = sortedMembers.slice(0, MAX_NODES_PER_CLUSTER);
            const visibleSet = new Set(visibleMembers.map(m => m.n));

            let edgeCount = cy.edges().filter(e => {{
                const d = e.data();
                return visibleSet.has(d.source) && visibleSet.has(d.target);
            }}).length;

            for (const m of visibleMembers) {{
                if (edgeCount >= MAX_EDGES_PER_CLUSTER) break;
                const gi = G[m.n];
                if (!gi) continue;
                for (const p of gi.p) {{
                    if (edgeCount >= MAX_EDGES_PER_CLUSTER) break;
                    if (p.j >= thresh && visibleSet.has(p.n)) {{
                        const eid = m.n < p.n ? `${{m.n}}||${{p.n}}` : `${{p.n}}||${{m.n}}`;
                        if (!graphEdges.has(eid)) {{
                            graphEdges.add(eid);
                            cy.add({{
                                group: 'edges',
                                data: {{
                                    id: eid,
                                    source: m.n < p.n ? m.n : p.n,
                                    target: m.n < p.n ? p.n : m.n,
                                    jaccard: p.j,
                                    source_name: m.n < p.n ? m.n : p.n,
                                    target_name: m.n < p.n ? p.n : m.n,
                                    width: 0.3 + p.j * 1.5,
                                    color: color,
                                }}
                            }});
                            edgeCount++;
                        }}
                    }}
                }}
            }}
        }});

        updateStatus();
        return;
    }}

    // 1. Build sidebar
    const activeSet = getActiveGeneSet();
    const filterText = geneFilterMode !== 'all' && document.getElementById('gene-filter') ? ` (filtered by ${{document.getElementById('gene-filter').selectedOptions[0].text}})` : '';
    document.getElementById('gene-info').innerHTML = `
        <h2>🔬 All Clusters <span style="font-size:11px;color:#8892b0;font-weight:400;">(Leiden)</span></h2>
        <div class="meta">
            <strong>${{nClusters}}</strong> clusters shown${{filterText}}<br>
            Click a cluster to zoom in • <strong>Click again</strong> (or double-click) to view all members.
        </div>
    `;

    let sideHtml = '';
    clusterIds.forEach((cid, i) => {{
        const size = getClusterMembers(cid).length;
        const color = clusterColors[cid] || '#38bdf8';
        const cname = CLUSTER_NAMES[cid] || `Cluster ${{cid}}`;
        const cilCount = (CLUSTERS[cid] || []).filter(name => ALL_CILIARY.has(name)).length;
        sideHtml += `
        <div class="cluster-entry" data-cluster-id="${{cid}}" ondblclick="showSingleCluster(${{cid}})" title="C${{cid}}: ${{cname}}">
            <span class="swatch" style="background:${{color}};"></span>
            <div style="flex:1; min-width:0; overflow:hidden;">
                <div class="cname" style="white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                    <strong style="color:#7eb8ff;">C${{cid}}</strong>: ${{cname}}
                </div>
            </div>
            ${{cilCount > 0 ? `<span class="cilia-badge" style="font-size:8px;padding:0 3px;" title="${{cilCount}} curated ciliary genes in cluster">${{cilCount}} cil</span>` : ''}}
            <span class="csize">${{size}} gene${{size !== 1 ? 's' : ''}}</span>
        </div>`;
    }});
    document.getElementById('partner-list').innerHTML = sideHtml;

    // Click handler for cluster entries (zoom on 1st click, open full cluster on 2nd click)
    document.querySelectorAll('.cluster-entry').forEach(el => {{
        el.addEventListener('click', function(e) {{
            if (e.detail >= 2) return; // skip double-click handled by ondblclick
            const cid = parseInt(this.dataset.clusterId);
            handleClusterClick(cid);
        }});
    }});

    // 2. Build graph
    cy.elements().remove();
    graphGenes.clear();
    graphEdges.clear();
    document.getElementById('empty-msg').style.display = 'none';

    // Compute grid layout for cluster centers
    const cols = Math.ceil(Math.sqrt(nClusters));
    const spacing = 600;  // spacing between cluster centers

    clusterIds.forEach((cid, idx) => {{
        const members = getClusterMembers(cid);
        const color = clusterColors[cid];
        const cname = CLUSTER_NAMES[cid] || `Cluster ${{cid}}`;
        const gridRow = Math.floor(idx / cols);
        const gridCol = idx % cols;
        const cx = gridCol * spacing;
        const cy_pos = gridRow * spacing;

        // Take top nodes by loss count (hub nodes first)
        const sortedMembers = members
            .filter(n => G[n])
            .map(n => ({{ n, l: G[n].l }}))
            .sort((a, b) => b.l - a.l);

        const visibleMembers = sortedMembers.slice(0, MAX_NODES_PER_CLUSTER);
        const visibleSet = new Set(visibleMembers.map(m => m.n));
        const hiddenCount = Math.max(0, sortedMembers.length - visibleMembers.length);

        // Place members in a small circle around the cluster center
        const nVis = visibleMembers.length;
        const radius = Math.max(40, Math.min(180, Math.sqrt(nVis) * 30));

        visibleMembers.forEach((m, mi) => {{
            const angle = (2 * Math.PI * mi) / nVis;
            const x = cx + radius * Math.cos(angle);
            const y = cy_pos + radius * Math.sin(angle);

            if (!graphGenes.has(m.n)) {{
                graphGenes.add(m.n);
                const showLabels = document.getElementById('toggle-labels') ? document.getElementById('toggle-labels').checked : true;
                const shortLabel = m.n.length > 12 ? m.n.slice(0, 10) + '…' : m.n;
                cy.add({{
                    group: 'nodes',
                    data: {{
                        id: m.n,
                        label: shortLabel,
                        fullName: m.n,
                        losses: m.l,
                        size: 5 + Math.sqrt(m.l) * 0.8,
                        color: color,
                        clusterId: cid,
                    }},
                    position: {{ x, y }},
                    classes: showLabels ? '' : 'hide-label',
                }});
            }}
        }});

        // Add a cluster label node (invisible hub for reference)
        const labelId = `__cluster_label_${{cid}}`;
        const shortName = cname.length > 26 ? cname.slice(0, 24) + '…' : cname;
        cy.add({{
            group: 'nodes',
            data: {{
                id: labelId,
                label: `C${{cid}}: ${{shortName}}\\n(${{members.length}} genes)` + (hiddenCount > 0 ? `\\n+${{hiddenCount}} more` : ''),
                fullName: `Cluster ${{cid}}: ${{cname}} (${{members.length}} genes)`,
                losses: 0,
                size: 1,
                color: 'transparent',
                clusterId: cid,
                isLabel: true,
            }},
            position: {{ x: cx, y: cy_pos }},
            classes: 'cluster-label',
        }});

        // Add top intra-cluster edges
        let edgeCount = 0;
        for (const m of visibleMembers) {{
            if (edgeCount >= MAX_EDGES_PER_CLUSTER) break;
            const gi = G[m.n];
            if (!gi) continue;
            for (const p of gi.p) {{
                if (edgeCount >= MAX_EDGES_PER_CLUSTER) break;
                if (p.j >= thresh && visibleSet.has(p.n)) {{
                    const eid = m.n < p.n ? `${{m.n}}||${{p.n}}` : `${{p.n}}||${{m.n}}`;
                    if (!graphEdges.has(eid)) {{
                        graphEdges.add(eid);
                        cy.add({{
                            group: 'edges',
                            data: {{
                                id: eid,
                                source: m.n < p.n ? m.n : p.n,
                                target: m.n < p.n ? p.n : m.n,
                                jaccard: p.j,
                                source_name: m.n < p.n ? m.n : p.n,
                                target_name: m.n < p.n ? p.n : m.n,
                                width: 0.3 + p.j * 1.5,
                                color: color,
                            }}
                        }});
                        edgeCount++;
                    }}
                }}
            }}
        }}
    }});

    cy.fit(undefined, 40);
    updateStatus();
    showToast(`Showing ${{nClusters}} Leiden clusters (${{cy.nodes().length}} nodes). Click a cluster in the sidebar to zoom, double-click for detail.`, 4000);
}}


function handleClusterClick(cid, highlightGene) {{
    if (lastClickedClusterId === cid) {{
        lastClickedClusterId = null;
        showSingleCluster(cid, highlightGene);
    }} else {{
        lastClickedClusterId = cid;
        zoomToCluster(cid);
    }}
}}

function zoomToCluster(cid) {{
    const clusterNodes = cy.nodes().filter(n => n.data('clusterId') === cid && !n.data('isLabel'));
    if (clusterNodes.length === 0) return;
    cy.animate({{
        fit: {{ eles: clusterNodes, padding: 80 }}
    }}, {{ duration: 400 }});

    // Highlight these nodes
    cy.nodes().removeClass('focus');
    clusterNodes.addClass('focus');

    // Update sidebar highlight and scroll into view
    document.querySelectorAll('.cluster-entry').forEach(el => {{
        const isCurrent = parseInt(el.dataset.clusterId) === cid;
        el.style.background = isCurrent ? '#1a2a4a' : '';
        if (isCurrent) el.scrollIntoView({{ block: 'nearest', behavior: 'smooth' }});
    }});
    const cname = CLUSTER_NAMES[cid] || `Cluster ${{cid}}`;
    showToast(`Zoomed to C${{cid}}: ${{cname}}. Click again to open full cluster.`, 3500);
}}

function showSingleCluster(cid, highlightGene, isFilterUpdate) {{
    if (!CLUSTERS[cid]) return;
    lastClickedClusterId = null;
    currentView = 'single_cluster';
    currentClusterId = cid;
    currentClusterHighlightGene = highlightGene || null;
    clusterViewActive = false;

    const thresh = parseFloat(document.getElementById('thresh').value);
    const topn = getTopN();
    const activeSet = getActiveGeneSet();
    const allMembers = (CLUSTERS[cid] || []).filter(n => G[n]);
    const color = clusterColors[cid] || 'hsl(200, 65%, 55%)';
    const cname = CLUSTER_NAMES[cid] || `Cluster ${{cid}}`;

    let members = [];
    let shownMembers = [];
    const bridgeSet = new Set();

    if (!activeSet) {{
        // All genes mode
        const sortedAll = allMembers.map(n => ({{ n, l: G[n].l, isCil: ALL_CILIARY.has(n), isHopper: false }})).sort((a, b) => b.l - a.l);
        members = sortedAll;
        shownMembers = sortedAll.slice(0, topn);
        if (highlightGene && !shownMembers.some(m => m.n === highlightGene)) {{
            const hlObj = sortedAll.find(m => m.n === highlightGene);
            if (hlObj) shownMembers.push(hlObj);
        }}
    }} else {{
        // Curated Ciliary Filter is ACTIVE:
        // Ciliary members
        const cilMembers = allMembers.filter(n => activeSet.has(n));
        const cilSet = new Set(cilMembers);

        // Find non-ciliary genes that connect >= 2 ciliary members (or connects highlightGene)
        const nonCilMembers = allMembers.filter(n => !activeSet.has(n));
        const bridgeMap = new Map(); // nonCilGene -> count of ciliary connections

        for (const u of nonCilMembers) {{
            const gi = G[u];
            if (!gi) continue;
            let nConnected = 0;
            for (const p of gi.p) {{
                if (p.j >= thresh && cilSet.has(p.n)) {{
                    nConnected++;
                }}
            }}
            const isHl = highlightGene && u === highlightGene;
            if (nConnected >= 2 || (isHl && nConnected >= 1)) {{
                bridgeMap.set(u, nConnected);
            }}
        }}

        const sortedCil = cilMembers.map(n => ({{ n, l: G[n].l, isCil: true, isBridge: false, nBridges: 0 }}))
            .sort((a, b) => b.l - a.l);

        // Direct cluster members are NOT hoppers (they are genuine cluster members like DRC9 in C6)
        // Only mark as hopper if the node is an external connector from outside the primary module
        const sortedBridges = [...bridgeMap.keys()].map(u => ({{
            n: u,
            l: G[u].l,
            isCil: false,
            isHopper: false, // Direct cluster member, not a hopper!
            nBridges: bridgeMap.get(u)
        }})).sort((a, b) => (b.nBridges - a.nBridges) || (b.l - a.l));

        members = [...sortedCil, ...sortedBridges];

        let shownCil = sortedCil.slice(0, topn);
        if (highlightGene && cilSet.has(highlightGene) && !shownCil.some(m => m.n === highlightGene)) {{
            const hlObj = sortedCil.find(m => m.n === highlightGene);
            if (hlObj) shownCil.push(hlObj);
        }}

        const remainingSlots = Math.max(0, topn - shownCil.length);
        let shownBridges = sortedBridges.slice(0, Math.max(remainingSlots, (highlightGene && bridgeMap.has(highlightGene)) ? 1 : 0));
        if (highlightGene && !cilSet.has(highlightGene) && bridgeMap.has(highlightGene) && !shownBridges.some(m => m.n === highlightGene)) {{
            const hlObj = sortedBridges.find(m => m.n === highlightGene);
            if (hlObj) shownBridges.push(hlObj);
        }}

        shownMembers = [...shownCil, ...shownBridges];
        sortedBridges.forEach(b => bridgeSet.add(b.n));
    }}

    const sorted = members;
    const shownSet = new Set(shownMembers.map(m => m.n));

    // Sidebar
    const hlEscaped = highlightGene ? highlightGene.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;') : '';
    const filterText = geneFilterMode !== 'all' && document.getElementById('gene-filter') ? ` (filtered by ${{document.getElementById('gene-filter').selectedOptions[0].text}})` : '';
    const nBridgesShown = shownMembers.filter(m => bridgeSet.has(m.n)).length;
    const bridgeNotice = nBridgesShown > 0 ? ` <span style="color:#94a3b8;">(+${{nBridgesShown}} dimmed bridge${{nBridgesShown !== 1 ? 's' : ''}})</span>` : '';
    const memberLabel = geneFilterMode !== 'all' ? 'ciliary members' : 'members';
    document.getElementById('gene-info').innerHTML = `
        <h2 style="color:${{color}}; font-size:16px;">C${{cid}}: ${{cname}}</h2>
        <div class="meta">
            <strong>${{shownMembers.length - nBridgesShown}}</strong> ${{memberLabel}} shown${{bridgeNotice}}${{filterText}} &nbsp;|&nbsp;
            <a href="#" onclick="showAllClusters(); return false;" style="color:#7eb8ff;text-decoration:underline;">← Back to all clusters</a>
            ${{shownMembers.length < members.length ? ' (increase "Show Top" to see more)' : ''}}
            ${{highlightGene ? `<br><span style="color:#5eff8a;">Focus gene: <strong>${{hlEscaped}}</strong></span>` : ''}}
        </div>
    `;

    let html = '';
    sorted.forEach((m, i) => {{
        const isHl = m.n === highlightGene;
        const inGraph = shownSet.has(m.n);
        const dispName = m.n.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
        const escaped = m.n.replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
        const isCil = ALL_CILIARY.has(m.n);
        const isBridge = bridgeSet.has(m.n);

        let rowStyle = '';
        if (isHl) {{
            rowStyle = 'background:#1a2a4a; border-left:3px solid #38bdf8;';
        }} else if (m.isHopper) {{
            rowStyle = inGraph ? 'opacity:0.65; background:rgba(71,85,105,0.18);' : 'opacity:0.32;';
        }} else if (!inGraph) {{
            rowStyle = 'opacity:0.45;';
        }}

        let badgeHtml = '';
        if (isCil) {{
            badgeHtml = getCiliaBadgeHtml(m.n);
        }} else if (m.isHopper) {{
            badgeHtml = `<span class="bridge-badge" title="Hopper: intermediate node bridging ${{m.nBridges || 2}} ciliary members">hopper (${{m.nBridges || 2}} cil)</span>`;
        }}

        html += `
        <div class="partner" data-gene="${{escaped}}" style="${{rowStyle}}">
            <span class="rank">#${{i + 1}}</span>
            <span class="pname ${{inGraph ? 'in-graph' : ''}}" style="${{isHl ? 'color:#38bdf8; font-weight:700;' : (m.isHopper ? 'color:#94a3b8;font-style:italic;' : '')}}">${{dispName}}</span>
            ${{badgeHtml}}
            <span class="ploss" style="width:auto;">${{m.l}}L</span>
            <a href="${{getUniProtUrl(m.n)}}" target="_blank" rel="noopener noreferrer" class="partner-ext-link" title="Open ${{dispName}} on UniProt" onclick="event.stopPropagation();">↗</a>
        </div>`;
    }});
    document.getElementById('partner-list').innerHTML = html;

    // In-place graph adjustment if isFilterUpdate and graph already exists
    if (isFilterUpdate && cy && cy.nodes().length > 0) {{
        // Remove nodes that are no longer in shownMembers
        cy.nodes().forEach(node => {{
            const id = node.data('id');
            if (!shownSet.has(id)) {{
                graphGenes.delete(id);
                cy.remove(node);
            }}
        }});

        // Add any newly included nodes in shownMembers (positioned near existing nodes)
        const refNode = highlightGene ? cy.getElementById(highlightGene) : cy.nodes()[0];
        const refPos = refNode && refNode.length ? refNode.position() : {{ x: 0, y: 0 }};

        shownMembers.forEach(m => {{
            if (!graphGenes.has(m.n)) {{
                const isHopper = m.isHopper;
                addGeneNode(m.n, m.n === highlightGene, isHopper);
                const node = cy.getElementById(m.n);
                if (node.length && m.n !== highlightGene && !isHopper) node.data('color', color);
                const angle = Math.random() * 2 * Math.PI;
                const dist = 60 + Math.random() * 120;
                node.position({{ x: refPos.x + Math.cos(angle) * dist, y: refPos.y + Math.sin(angle) * dist }});
            }}
        }});

        // Remove edges that fall below threshold or connect removed nodes
        cy.edges().forEach(edge => {{
            const d = edge.data();
            if (d.jaccard < thresh || !shownSet.has(d.source) || !shownSet.has(d.target)) {{
                graphEdges.delete(d.id);
                cy.remove(edge);
            }}
        }});

        // Add qualifying edges that are not yet drawn
        shownMembers.forEach(m => {{
            const gi = G[m.n];
            if (!gi) return;
            for (const p of gi.p) {{
                if (p.j >= thresh && shownSet.has(p.n)) {{
                    const isBridgeEdge = bridgeSet.has(m.n) || bridgeSet.has(p.n);
                    addEdge(m.n, p.n, p.j, isBridgeEdge);
                }}
            }}
        }});

        updateStatus();
        return;
    }}

    // Initial full render
    cy.elements().remove();
    graphGenes.clear();
    graphEdges.clear();
    document.getElementById('empty-msg').style.display = 'none';

    for (const m of shownMembers) {{
        const isHopper = m.isHopper;
        addGeneNode(m.n, m.n === highlightGene, isHopper);
        const node = cy.getElementById(m.n);
        if (node.length && m.n !== highlightGene && !isHopper) node.data('color', color);
    }}

    for (const m of shownMembers) {{
        const gi = G[m.n];
        if (!gi) continue;
        for (const p of gi.p) {{
            if (p.j >= thresh && shownSet.has(p.n)) {{
                const isHopperEdge = m.isHopper || (shownMembers.find(x => x.n === p.n) && shownMembers.find(x => x.n === p.n).isHopper);
                addEdge(m.n, p.n, p.j, isHopperEdge);
            }}
        }}
    }}

    runLayout();
    if (highlightGene && !isFilterUpdate) {{
        const hNode = cy.getElementById(highlightGene);
        if (hNode.length) {{
            hNode.select();
            setTimeout(() => {{
                cy.animate({{ fit: {{ eles: hNode, padding: 100 }} }}, {{ duration: 400 }});
            }}, 500);
        }}
    }}
    updateStatus();
}}

// Debounced filter handler: adjusts in-place immediately, then refreshes layout after 1s of inactivity
let debounceTimer = null;
let layoutRefreshTimer = null;

function onFilterChange() {{
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {{
        if (currentView === 'single_cluster' && currentClusterId !== null) {{
            showSingleCluster(currentClusterId, currentClusterHighlightGene, true);
        }} else if (currentView === 'all_clusters') {{
            showAllClusters(true);
        }} else if (selectedGene) {{
            renderNetwork(selectedGene);
        }}
    }}, 80);

    // After 1 second of no further filter changes, refresh layout
    clearTimeout(layoutRefreshTimer);
    layoutRefreshTimer = setTimeout(() => {{
        if (currentView === 'single_cluster' || currentView === 'gene') {{
            runLayout(false);
        }}
    }}, 1000);
}}

document.getElementById('thresh').addEventListener('input', function() {{
    document.getElementById('thresh-val').textContent = parseFloat(this.value).toFixed(2);
    onFilterChange();
}});

document.getElementById('topn').addEventListener('input', function() {{
    document.getElementById('topn-val').textContent = this.value;
    onFilterChange();
}});

document.getElementById('toggle-topn-max').addEventListener('change', function() {{
    const topnSlider = document.getElementById('topn');
    topnSlider.disabled = this.checked;
    document.getElementById('topn-val').textContent = this.checked ? 'Max' : topnSlider.value;
    onFilterChange();
}});

document.getElementById('gene-filter').addEventListener('change', function() {{
    geneFilterMode = this.value;
    if (currentView === 'single_cluster' && currentClusterId !== null) {{
        showSingleCluster(currentClusterId, currentClusterHighlightGene, false);
    }} else if (currentView === 'all_clusters') {{
        showAllClusters(false);
    }} else if (currentView === 'gene' && selectedGene) {{
        renderNetwork(selectedGene);
    }}
}});


document.getElementById('layout-select').addEventListener('change', function() {{
    runLayout();
}});

document.getElementById('toggle-labels').addEventListener('change', function() {{
    const show = this.checked;
    cy.nodes().forEach(n => {{
        if (n.data('id') !== selectedGene) {{
            n.toggleClass('hide-label', !show);
        }}
    }});
}});

// Export button
function exportCytoscape() {{
    if (!cy) return;
    const jsonStr = JSON.stringify(cy.json());
    const blob = new Blob([jsonStr], {{ type: 'application/json' }});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'network.json';
    a.click();
    URL.revokeObjectURL(url);
}}




// ---- Status Bar ----
function updateStatus() {{
    if (!cy) return;
    document.getElementById('status-nodes').textContent = `Nodes: ${{cy.nodes().length}}`;
    document.getElementById('status-edges').textContent = `Edges: ${{cy.edges().length}}`;
}}

// ---- Search & Autocomplete ----

let sugIdx = -1;
let sugItems = [];

searchInput.addEventListener('input', function() {{
    const q = this.value.trim().toUpperCase();
    sugIdx = -1;
    if (q.length === 0) {{
        sugBox.style.display = 'none';
        return;
    }}
    // Exact prefix matches first, then contains
    const prefix = NAMES.filter(n => n.toUpperCase().startsWith(q));
    const contains = NAMES.filter(n => !n.toUpperCase().startsWith(q) && n.toUpperCase().includes(q));
    sugItems = [...prefix, ...contains].slice(0, 20);
    if (sugItems.length === 0) {{
        sugBox.style.display = 'none';
        return;
    }}
    sugBox.innerHTML = sugItems.map((n, i) => {{
        const escaped = n.replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
        const dispName = n.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
        return `<div data-gene="${{escaped}}" ${{i === sugIdx ? 'class="active"' : ''}}>${{dispName}} <span style="color:#556;font-size:11px;">${{G[n] ? G[n].l + 'L' : ''}}</span></div>`;
    }}).join('');
    sugBox.style.display = 'block';
}});

searchInput.addEventListener('keydown', function(e) {{
    if (e.key === 'ArrowDown') {{
        e.preventDefault();
        sugIdx = Math.min(sugIdx + 1, sugItems.length - 1);
        highlightSug();
    }} else if (e.key === 'ArrowUp') {{
        e.preventDefault();
        sugIdx = Math.max(sugIdx - 1, 0);
        highlightSug();
    }} else if (e.key === 'Enter') {{
        e.preventDefault();
        const pick = sugIdx >= 0 ? sugItems[sugIdx] : sugItems[0];
        if (pick) pickSuggestion(pick);
    }} else if (e.key === 'Escape') {{
        sugBox.style.display = 'none';
    }}
}});

function highlightSug() {{
    const divs = sugBox.querySelectorAll('div');
    divs.forEach((d, i) => d.classList.toggle('active', i === sugIdx));
}}

function pickSuggestion(name) {{
    sugBox.style.display = 'none';
    searchInput.value = name;
    renderNetwork(name);
}}

searchInput.addEventListener('blur', () => {{
    setTimeout(() => sugBox.style.display = 'none', 200);
}});

// ---- Event delegation for data-gene clicks ----
document.getElementById('partner-list').addEventListener('click', function(e) {{
    const row = e.target.closest('.partner[data-gene]');
    if (row && row.dataset.gene) renderNetwork(row.dataset.gene);
}});

document.getElementById('suggestions').addEventListener('mousedown', function(e) {{
    // mousedown instead of click so it fires before the blur hides the dropdown
    const item = e.target.closest('[data-gene]');
    if (item) {{
        e.preventDefault();
        pickSuggestion(item.dataset.gene);
    }}
}});

// ---- Legend Toggle ----
document.getElementById('legend-toggle').addEventListener('click', function(e) {{
    e.stopPropagation();
    const body = document.getElementById('legend-body');
    const isHidden = body.style.display === 'none';
    body.style.display = isHidden ? 'block' : 'none';
    this.textContent = isHidden ? '−' : '+';
}});

// ---- Init ----
function startExplorer() {{
    initCy();
    if (!HAS_CLUSTERS) {{
        const btn = document.getElementById('btn-all-clusters');
        if (btn) btn.style.display = 'none';
        const headBtn = document.getElementById('btn-all-clusters-head');
        if (headBtn) headBtn.style.display = 'none';
    }}
}}

if (document.readyState === 'loading') {{
    document.addEventListener('DOMContentLoaded', startExplorer);
}} else {{
    startExplorer();
}}



</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
