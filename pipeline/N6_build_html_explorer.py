#!/usr/bin/env python3
"""
Build a self-contained HTML network explorer with Dual-Engine architecture:
1. Sigma.js (WebGL): For the whole network (11,236 genes, 73,467 edges) with precomputed 2D layout.
2. Cytoscape.js (Canvas): For focused gene networks, multi-hop BFS, and cluster exploration.
3. Binary Streaming: Uses network_whole.bin (518 KB) and network_partners.bin (4.2 MB) for instant zero-lag loading.

Usage:
    python3 build_html_explorer.py [--top-k 100] [--min-loss 5] [--jaccard-floor 0.05]
"""

import argparse
import csv
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import igraph as ig


def hsl_to_hex(h, s, l):
    s /= 100.0
    l /= 100.0
    a = s * min(l, 1.0 - l)
    def f(n):
        k = (n + h / 30.0) % 12.0
        color = l - a * max(min(k - 3.0, 9.0 - k, 1.0), -1.0)
        return f"{int(round(255 * color)):02x}"
    return f"#{f(0)}{f(8)}{f(4)}"


def main():
    parser = argparse.ArgumentParser(description="Build interactive Dual-Engine HTML network explorer")
    parser.add_argument("--top-k", type=int, default=100,
                        help="Max partners to store per gene in network_partners.bin (default: 100)")
    parser.add_argument("--min-loss", type=int, default=5,
                        help="Minimum losses for a gene to be included (default: 5)")
    parser.add_argument("--jaccard-floor", type=float, default=0.05,
                        help="Minimum Jaccard to include as partner (default: 0.05)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output HTML file path (default: ./index.html)")
    args = parser.parse_args()

    website_dir = Path(__file__).resolve().parent
    base = website_dir
    results = base / "results"

    print("1. Loading gene and matrix data...")
    J = np.load(results / "jaccard_matrix.npy", mmap_mode="r")
    loss_counts = np.load(results / "loss_counts.npy")

    genes = []
    with open(results / "jaccard_genes.tsv") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            genes.append({"idx": int(parts[0]), "fid": parts[1], "name": parts[2]})

    eligible = [g for g in genes if loss_counts[g["idx"]] >= args.min_loss]
    all_names = sorted([g["name"] for g in eligible])
    name_to_id = {name: i for i, name in enumerate(all_names)}
    name_to_orig_idx = {g["name"]: g["idx"] for g in genes}
    idx_to_name = {g["idx"]: g["name"] for g in genes}
    n_genes = len(all_names)
    print(f"  {n_genes} eligible genes (losses >= {args.min_loss})")

    # Load Leiden clusters
    cluster_data = {}
    gene_to_cluster = {}
    leiden_path = results / "leiden_clusters.tsv"
    if leiden_path.exists():
        print("Loading Leiden cluster assignments...")
        from collections import defaultdict
        clusters_raw = defaultdict(list)
        with open(leiden_path) as f:
            next(f)
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    gname, cid = parts[0], int(parts[1])
                    if gname in name_to_id:
                        clusters_raw[cid].append(gname)
                        gene_to_cluster[gname] = cid
        cluster_data = dict(clusters_raw)
        print(f"  {len(cluster_data)} clusters loaded")

    # Annotate clusters with GO
    cluster_names = {}
    if cluster_data:
        cluster_names = annotate_clusters_with_go(cluster_data, base)
        print(f"  Annotated {len(cluster_names)} clusters with GO terms")

    # Precompute cluster hex colors (Sigma.js requires HEX, not HSL!)
    cluster_colors = {}
    sorted_cids = sorted(cluster_data.keys(), key=lambda cid: len(cluster_data[cid]), reverse=True)
    n_cl = max(1, len(sorted_cids))
    for i, cid in enumerate(sorted_cids):
        h = (i * 360.0 / n_cl + 15.0) % 360.0
        cluster_colors[cid] = hsl_to_hex(h, 75, 60)

    # Load curated ciliary gene sets
    ciliary_sets, cilia_info = load_ciliary_annotations(base)

    # 2. Build or verify network_whole.bin (Whole Network)
    whole_bin_path = website_dir / "network_whole.bin"
    edges_tsv_path = website_dir / "network_edges.tsv"
    
    if not whole_bin_path.exists() and edges_tsv_path.exists():
        print("2. Generating network_whole.bin (Precomputing DrL layout)...")
        generate_whole_binary(whole_bin_path, edges_tsv_path, all_names, name_to_id, 
                              name_to_orig_idx, loss_counts, gene_to_cluster)
    else:
        print(f"2. Using existing {whole_bin_path}")

    # 3. Build or verify network_partners.bin (Partner lists for Ego/Pathways)
    partners_bin_path = website_dir / "network_partners.bin"
    if not partners_bin_path.exists():
        print("3. Generating network_partners.bin...")
        generate_partners_binary(partners_bin_path, J, all_names, name_to_id,
                                 name_to_orig_idx, idx_to_name, loss_counts,
                                 args.top_k, args.jaccard_floor)
    else:
        print(f"3. Using existing {partners_bin_path}")

    # 4. Generate index.html
    print("4. Generating index.html...")
    output_path = Path(args.output or str(website_dir / "index.html"))
    html = build_html(all_names, cluster_data, cluster_names, cluster_colors, ciliary_sets, cilia_info)

    with open(output_path, "w") as f:
        f.write(html)

    size_kb = output_path.stat().st_size / 1024
    print(f"Successfully generated {output_path} ({size_kb:.1f} KB)")


def generate_whole_binary(bin_path, edges_path, all_names, name_to_id, 
                          name_to_orig_idx, loss_counts, gene_to_cluster):
    edges_df = pd.read_csv(edges_path, sep="\t")
    valid_mask = edges_df["gene1"].isin(name_to_id) & edges_df["gene2"].isin(name_to_id)
    valid_edges = edges_df[valid_mask]
    n_genes = len(all_names)

    edge_list = [(name_to_id[g1], name_to_id[g2]) for g1, g2 in zip(valid_edges["gene1"], valid_edges["gene2"])]
    weights = valid_edges["jaccard"].tolist()

    g = ig.Graph(n=n_genes, edges=edge_list, directed=False)
    g.es["weight"] = weights

    layout = g.layout_drl(weights="weight", dim=2)
    coords = np.array(layout.coords)

    min_c = coords.min(axis=0)
    max_c = coords.max(axis=0)
    norm_coords = ((coords - min_c) / (max_c - min_c + 1e-6) - 0.5) * 4000

    header_whole = struct.pack("<4sII", b"DLW1", n_genes, len(edge_list))
    node_bytes = bytearray()
    for i, name in enumerate(all_names):
        orig_idx = name_to_orig_idx[name]
        x_val = int(round(norm_coords[i, 0]))
        y_val = int(round(norm_coords[i, 1]))
        losses = int(loss_counts[orig_idx])
        cid = gene_to_cluster.get(name, 65535)
        node_bytes.extend(struct.pack("<hhHH", x_val, y_val, losses, cid))

    edge_bytes = bytearray()
    for (u, v), j_val in zip(edge_list, weights):
        j_u16 = max(0, min(65535, int(round(j_val * 65535))))
        edge_bytes.extend(struct.pack("<HHH", u, v, j_u16))

    with open(bin_path, "wb") as f:
        f.write(header_whole)
        f.write(node_bytes)
        f.write(edge_bytes)
    sz = len(header_whole) + len(node_bytes) + len(edge_bytes)
    print(f"  Wrote {bin_path} ({sz/1024:.1f} KB)")


def generate_partners_binary(bin_path, J, all_names, name_to_id,
                             name_to_orig_idx, idx_to_name, loss_counts,
                             top_k, jaccard_floor):
    n_genes = len(all_names)
    partner_bytes = bytearray()
    offsets = []
    gene_losses = []

    for gid, name in enumerate(all_names):
        orig_idx = name_to_orig_idx[name]
        row = np.array(J[orig_idx, :], dtype=np.float32)
        row[orig_idx] = 0.0
        losses = int(loss_counts[orig_idx])
        gene_losses.append(losses)
        
        offsets.append(len(partner_bytes))
        
        above_floor = np.where(row >= jaccard_floor)[0]
        if len(above_floor) == 0:
            partner_bytes.extend(struct.pack("<H", 0))
            continue
            
        scores = row[above_floor]
        if len(scores) > top_k:
            topk_local = np.argpartition(scores, -top_k)[-top_k:]
            topk_indices = above_floor[topk_local]
            topk_scores = scores[topk_local]
        else:
            topk_indices = above_floor
            topk_scores = scores
            
        order = np.argsort(-topk_scores)
        partners = []
        for o in order:
            pidx = int(topk_indices[o])
            pname = idx_to_name.get(pidx)
            if pname is not None and pname in name_to_id:
                j_val = float(topk_scores[o])
                j_u16 = max(0, min(65535, int(round(j_val * 65535))))
                partners.append((name_to_id[pname], j_u16))
                
        partner_bytes.extend(struct.pack("<H", len(partners)))
        for pid, j_u16 in partners:
            partner_bytes.extend(struct.pack("<HH", pid, j_u16))

    offsets.append(len(partner_bytes))

    header_partners = struct.pack("<4sI", b"DLP1", n_genes)
    offset_bytes = struct.pack(f"<{len(offsets)}I", *offsets)
    loss_bytes = struct.pack(f"<{len(gene_losses)}H", *gene_losses)

    with open(bin_path, "wb") as f:
        f.write(header_partners)
        f.write(offset_bytes)
        f.write(loss_bytes)
        f.write(partner_bytes)

    sz = len(header_partners) + len(offset_bytes) + len(loss_bytes) + len(partner_bytes)
    print(f"  Wrote {bin_path} ({sz/(1024*1024):.2f} MB)")


def load_ciliary_annotations(base):
    data_roots = [base, base.parent]
    ciliary_genes_path = next((root / "ciliary_genes.csv" for root in data_roots if (root / "ciliary_genes.csv").exists()), None)
    ciliacarta_path = next((root / "CiliaCarta.csv" for root in data_roots if (root / "CiliaCarta.csv").exists()), None)

    cc_genes = set()
    if ciliacarta_path and ciliacarta_path.exists():
        with open(ciliacarta_path, newline="") as f:
            for row in csv.DictReader(f):
                sym = (row.get("Associated Gene Name") or "").strip()
                if sym:
                    cc_genes.add(sym)

    v2_genes = set()
    v1_genes = set()
    cilia_info = {}

    if ciliary_genes_path and ciliary_genes_path.exists():
        with open(ciliary_genes_path, newline="") as f:
            for row in csv.DictReader(f):
                gene = (row.get("Gene Name") or "").strip()
                if not gene:
                    continue
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
                    "cc": gene in cc_genes,
                    "loc": loc
                }

    for gene in cc_genes:
        if gene not in cilia_info:
            cilia_info[gene] = {"v2": "", "cc": True, "loc": ""}

    all_ciliary = v2_genes | cc_genes
    ciliary_sets = {
        "syscilia_v2": sorted(v2_genes),
        "ciliacarta": sorted(cc_genes),
        "ciliacarta_full": sorted(cc_genes),
        "shared_core": sorted(v2_genes & cc_genes),
        "both": sorted(all_ciliary),
        "all_ciliary": sorted(all_ciliary),
        "syscilia": sorted(v2_genes),
    }
    return ciliary_sets, cilia_info


def annotate_clusters_with_go(cluster_data, base_dir):
    import pickle
    from collections import Counter
    import scipy.stats as stats

    candidates = [
        base_dir / "data" / "go",
        base_dir.parent / "junk" / "coevolution_framework" / "helpers" / "go_data",
        base_dir / "junk" / "coevolution_framework" / "helpers" / "go_data",
    ]
    go_dir = next((c for c in candidates if (c / "symbol_go_map.pkl").exists() and (c / "go-basic.obo").exists()), None)

    if not go_dir:
        return {cid: f"Cluster {cid}" for cid in cluster_data}

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
                picked.append(term)
                if len(picked) == 2:
                    break
            cluster_names[cid] = " / ".join(picked)
        else:
            cluster_names[cid] = f"Cluster {cid}"

    return cluster_names


def build_html(all_names, cluster_data=None, cluster_names=None, cluster_colors=None, ciliary_sets=None, cilia_info=None):
    names_json = json.dumps(all_names, separators=(",", ":"))
    clusters_json = json.dumps(cluster_data or {}, separators=(",", ":"))
    cluster_names_json = json.dumps(cluster_names or {}, separators=(",", ":"))
    cluster_colors_json = json.dumps(cluster_colors or {}, separators=(",", ":"))
    ciliary_sets_json = json.dumps(ciliary_sets or {}, separators=(",", ":"))
    cilia_info_json = json.dumps(cilia_info or {}, separators=(",", ":"))

    html_code = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Gene Loss-Concordance Network Explorer</title>
<!-- Cytoscape for Focused Gene & Pathway Views -->
<script src="https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
<!-- Graphology & Sigma.js for 60 FPS WebGL Whole Graph View -->
<script src="https://cdn.jsdelivr.net/npm/graphology@0.25.4/dist/graphology.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/sigma@2.4.0/build/sigma.min.js"></script>
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
    padding: 8px 16px;
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
    z-index: 10;
}}
#header h1 {{
    font-size: 15px;
    font-weight: 600;
    color: #7eb8ff;
    white-space: nowrap;
}}
.view-toggle {{
    display: inline-flex;
    background: #0d1220;
    border: 1px solid #3a4570;
    border-radius: 6px;
    padding: 2px;
    gap: 2px;
}}
.view-btn {{
    padding: 5px 12px;
    font-size: 12px;
    font-weight: 600;
    border: none;
    border-radius: 4px;
    background: transparent;
    color: #8892b0;
    cursor: pointer;
    transition: all 0.15s ease;
    display: inline-flex;
    align-items: center;
    gap: 5px;
}}
.view-btn:hover {{
    color: #e2e8f0;
}}
.view-btn.active {{
    background: #2563eb;
    color: #ffffff;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
}}
.btn-clusters-header {{
    background: #1e293b;
    border: 1px solid #38bdf8;
    color: #38bdf8;
    border-radius: 6px;
    padding: 5px 12px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
    display: inline-flex;
    align-items: center;
    gap: 5px;
}}
.btn-clusters-header:hover, .btn-clusters-header.active {{
    background: #0284c7;
    border-color: #0284c7;
    color: #ffffff;
}}
.search-box {{
    position: relative;
    width: 250px;
}}
.search-box input {{
    width: 100%;
    padding: 5px 10px;
    border: 1px solid #3a4570;
    border-radius: 6px;
    background: #0d1220;
    color: #e0e6f0;
    font-size: 12px;
    outline: none;
}}
.search-box input:focus {{ border-color: #5a8eff; }}
#suggestions {{
    position: absolute;
    top: 100%;
    left: 0;
    right: 0;
    background: #151d30;
    border: 1px solid #2a3558;
    border-radius: 0 0 6px 6px;
    max-height: 250px;
    overflow-y: auto;
    z-index: 100;
    display: none;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
}}
#suggestions div {{
    padding: 6px 10px;
    cursor: pointer;
    font-size: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
#suggestions div:hover {{ background: #1f2b48; }}
.controls {{
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 12px;
    color: #8892b0;
    margin-left: auto;
    flex-wrap: wrap;
}}
.controls label {{ display: flex; align-items: center; gap: 4px; }}
.controls input[type="range"] {{ width: 80px; accent-color: #5a8eff; }}
.controls select {{
    background: #0d1220;
    border: 1px solid #3a4570;
    color: #e0e6f0;
    padding: 3px 6px;
    border-radius: 4px;
    font-size: 11px;
}}
.btn {{
    padding: 4px 10px;
    border: 1px solid #3a4570;
    border-radius: 4px;
    background: #1a2540;
    color: #c0d0e8;
    cursor: pointer;
    font-size: 11px;
    font-weight: 500;
    transition: all 0.15s;
}}
.btn:hover {{ background: #253555; color: #fff; }}
.btn-accent {{
    background: #1e3a5f;
    border-color: #38bdf8;
    color: #38bdf8;
}}
.btn-accent:hover {{
    background: #38bdf8;
    color: #0b1120;
}}
#main {{
    flex: 1;
    display: flex;
    position: relative;
    overflow: hidden;
}}
#graph-wrapper {{
    flex: 1;
    position: relative;
    background: #0a0e17;
    overflow: hidden;
}}
#sigma-container {{
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
}}
#cy {{
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    display: none;
}}
#webgl-hud {{
    position: absolute;
    top: 12px;
    left: 12px;
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(56, 189, 248, 0.3);
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 12px;
    color: #e2e8f0;
    pointer-events: none;
    backdrop-filter: blur(4px);
    z-index: 5;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
}}
#tooltip {{
    position: absolute;
    background: rgba(15, 23, 42, 0.95);
    border: 1px solid #38bdf8;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 11.5px;
    color: #f8fafc;
    pointer-events: none;
    z-index: 50;
    display: none;
    box-shadow: 0 4px 12px rgba(0,0,0,0.6);
    line-height: 1.4;
}}
#sidebar {{
    width: 370px;
    background: #111827;
    border-left: 1px solid #2a3050;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    flex-shrink: 0;
    z-index: 5;
}}
#sidebar-search-wrap {{
    position: relative;
    padding: 7px 10px;
    border-bottom: 1px solid #2a3050;
    background: #111827;
    flex-shrink: 0;
}}
#sidebar-search {{
    width: 100%;
    padding: 5px 26px 5px 26px;
    border: 1px solid #3a4570;
    border-radius: 5px;
    background: #0d1220;
    color: #e0e6f0;
    font-size: 12px;
    outline: none;
}}
#sidebar-search:focus {{ border-color: #5a8eff; }}
#sidebar-search-clear {{
    position: absolute;
    right: 18px;
    top: 50%;
    transform: translateY(-50%);
    color: #6b7280;
    cursor: pointer;
    font-size: 12px;
    display: none;
}}
#gene-info {{
    padding: 12px 16px;
    border-bottom: 1px solid #2a3050;
    background: #151d30;
    flex-shrink: 0;
}}
#gene-info h2 {{
    font-size: 16px;
    color: #7eb8ff;
    margin-bottom: 3px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}}
#gene-info .meta {{
    font-size: 11.5px;
    color: #8892b0;
    line-height: 1.4;
}}
#partner-list {{
    flex: 1;
    overflow-y: auto;
}}
.partner {{
    display: flex;
    align-items: center;
    padding: 6px 12px;
    border-bottom: 1px solid #1a2040;
    font-size: 12.5px;
    cursor: pointer;
    transition: background 0.1s;
    gap: 8px;
}}
.partner:hover {{ background: #1a2a4a; }}
.partner.active {{ background: #243560; }}
.partner .rank {{
    width: 26px;
    color: #556;
    font-size: 11px;
    font-tabular-nums: tabular-nums;
}}
.partner .pname {{
    flex: 1;
    font-weight: 500;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}}
.partner .pname.in-graph {{ color: #7eb8ff; }}
.partner .bar-wrap {{
    width: 50px;
    height: 6px;
    background: #1a2040;
    border-radius: 3px;
    overflow: hidden;
}}
.partner .bar {{
    height: 100%;
    background: #5a8eff;
    border-radius: 3px;
}}
.partner .pjaccard {{
    width: 38px;
    text-align: right;
    font-size: 11.5px;
    color: #8892b0;
}}
.partner .ploss {{
    width: 30px;
    text-align: right;
    font-size: 11px;
    color: #556;
}}
.cluster-entry {{
    display: flex;
    align-items: center;
    padding: 7px 12px;
    border-bottom: 1px solid #1a2040;
    font-size: 12px;
    cursor: pointer;
    transition: background 0.1s;
    gap: 8px;
}}
.cluster-entry:hover {{ background: #1a2a4a; }}
.cluster-entry.active {{ background: #243560; border-left: 3px solid #38bdf8; }}
.cluster-entry .swatch {{
    width: 12px;
    height: 12px;
    border-radius: 3px;
    flex-shrink: 0;
    border: 1px solid rgba(255,255,255,0.2);
}}
.cluster-entry .cname {{
    flex: 1;
    min-width: 0;
    font-weight: 500;
    color: #e0e6f0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}}
.cluster-entry .csize {{
    font-size: 11px;
    color: #8892b0;
    flex-shrink: 0;
}}
.cilia-badge {{
    font-size: 9px;
    padding: 1px 5px;
    border-radius: 3px;
    background: rgba(56, 189, 248, 0.2);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.4);
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    flex-shrink: 0;
}}
.gene-ext-link, .partner-ext-link {{
    font-size: 11px;
    color: #64748b;
    text-decoration: none;
    padding: 1px 5px;
    border-radius: 3px;
    border: 1px solid #2a3558;
    background: #0f172a;
    transition: all 0.15s;
    flex-shrink: 0;
}}
.gene-ext-link:hover, .partner-ext-link:hover {{
    color: #38bdf8;
    border-color: #38bdf8;
    background: #1e293b;
}}
.back-to-clusters {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
    color: #38bdf8;
    font-size: 11px;
    font-weight: 600;
    text-decoration: none;
    margin-bottom: 6px;
    cursor: pointer;
}}
.back-to-clusters:hover {{
    color: #7dd3fc;
    text-decoration: underline;
}}
#status-bar {{
    background: #0d1220;
    border-top: 1px solid #2a3050;
    padding: 4px 16px;
    font-size: 11px;
    color: #8892b0;
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
#loading-pill {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #1a2540;
    color: #7eb8ff;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
}}
.spinner {{
    width: 10px;
    height: 10px;
    border: 2px solid #38bdf8;
    border-top-color: transparent;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
}}
@keyframes spin {{
    to {{ transform: rotate(360deg); }}
}}
</style>
</head>
<body>

<div id="header">
    <h1>Dollo Co-Loss Network Explorer</h1>
    <div class="view-toggle">
        <button id="btn-view-whole" class="view-btn active" onclick="switchView('whole')">🌐 Whole Network (WebGL)</button>
        <button id="btn-view-cy" class="view-btn" onclick="switchView('gene')">🔬 Gene Focus (Cytoscape)</button>
    </div>
    <button id="btn-all-clusters-head" class="btn-clusters-header" onclick="showAllClusters()">🗂️ All Clusters (Leiden)</button>

    <div class="search-box">
        <input type="text" id="search" placeholder="Search gene (e.g. SCAPER, CEP290)…" autocomplete="off">
        <div id="suggestions"></div>
    </div>

    <div class="controls">
        <label>Jaccard &ge; <span id="thresh-val">0.20</span></label>
        <input type="range" id="thresh" min="0.10" max="0.70" step="0.02" value="0.20">

        <label>Show Top <span id="topn-val">25</span></label>
        <input type="range" id="topn" min="5" max="100" step="5" value="25">
        <label><input type="checkbox" id="toggle-topn-max"> Max</label>

        <label>Filter:</label>
        <select id="gene-filter">
            <option value="all">All Genes</option>
            <option value="all_ciliary">Curated Ciliary Genes</option>
            <option value="syscilia">SYSCILIA Gold Standard</option>
        </select>

        <label style="margin-left:4px;">Labels:</label>
        <input type="checkbox" id="toggle-labels" checked>
    </div>
</div>

<div id="main">
    <div id="graph-wrapper">
        <div id="sigma-container"></div>
        <div id="cy"></div>
        <div id="webgl-hud">
            <strong>WebGL View:</strong> 11,236 genes • 73,467 co-loss edges<br>
            <span style="color:#8892b0; font-size:11px;">Drag to pan • Scroll to zoom • Click node to focus</span>
        </div>
        <div id="empty-msg" style="position:absolute; inset:0; display:flex; align-items:center; justify-content:center; color:#556; font-size:14px; pointer-events:none; display:none;">
            Search for a gene or pick a cluster from the sidebar
        </div>
        <div id="tooltip"></div>
    </div>

    <div id="sidebar">
        <div id="sidebar-search-wrap">
            <span style="position:absolute; left:18px; top:50%; transform:translateY(-50%); color:#4a5568; font-size:12px;">🔍</span>
            <input type="text" id="sidebar-search" placeholder="Filter sidebar genes / clusters..." autocomplete="off">
            <span id="sidebar-search-clear">✕</span>
        </div>
        <div id="gene-info">
            <h2>Select a Gene</h2>
            <div class="meta">Search above or click any node in the network to inspect co-loss partners.</div>
        </div>
        <div id="partner-list"></div>
    </div>
</div>

<div id="status-bar">
    <span id="graph-status">Loading network data…</span>
    <div id="loading-pill"><div class="spinner"></div> Loading binary buffers…</div>
</div>

<script>
// ---- Static Inlined Metadata ----
const NAMES = {names_json};
const CLUSTERS = {clusters_json};
const CLUSTER_NAMES = {cluster_names_json};
const CLUSTER_COLORS = {cluster_colors_json};
const CILIARY_SETS = {ciliary_sets_json};
const CILIA_INFO = {cilia_info_json};
const ALL_CILIARY = new Set(CILIARY_SETS["all_ciliary"] || CILIARY_SETS["both"] || []);
const HAS_CLUSTERS = Object.keys(CLUSTERS).length > 0;

// Derived lookup indexes (< 2ms)
const GENE_IDX = {{}};
NAMES.forEach((name, i) => {{ GENE_IDX[name] = i; }});

const GENE_CL = {{}};
for (const cid in CLUSTERS) {{
    for (const name of CLUSTERS[cid]) {{
        GENE_CL[name] = Number(cid);
    }}
}}

// Fallback cluster color helper (Hex format for WebGL)
function getClusterColor(cid) {{
    if (cid !== undefined && CLUSTER_COLORS[cid]) return CLUSTER_COLORS[cid];
    return '#38bdf8';
}}

// ---- Global State ----
let currentMode = 'whole'; // 'whole' | 'gene' | 'all_clusters'
let selectedGene = null;
let currentClusterId = null;
let geneFilterMode = 'all';

let sigmaGraph = null;
let sigmaRenderer = null;
let cy = null;

// Binary buffers
let WHOLE_LOADED = false;
let PARTNERS_LOADED = false;
let GM = {{}}; // loss counts
let PARTNER_OFFSETS = null;
let PARTNER_LOSSES = null;
let PARTNER_DATA = null;
const G_CACHE = {{}};
let CLUSTER_CENTROIDS = {{}};

function isWebGLAvailable() {{
    try {{
        const canvas = document.createElement('canvas');
        return !!(window.WebGLRenderingContext && (canvas.getContext('webgl') || canvas.getContext('experimental-webgl')));
    }} catch (e) {{
        return false;
    }}
}}

function getUniProtUrl(gene) {{
    if (!gene) return 'https://www.uniprot.org';
    const trimmed = gene.trim();
    return `https://www.uniprot.org/uniprotkb?query=gene_exact:${{encodeURIComponent(trimmed)}}+AND+organism_id:9606`;
}}

function getCiliaBadgeHtml(geneName) {{
    if (!ALL_CILIARY.has(geneName)) return '';
    const info = CILIA_INFO[geneName];
    let title = 'Curated ciliary component';
    if (info && info.v2) title += ` (SYSCILIA ${{info.v2}})`;
    if (info && info.loc) title += `\nLocalization: ${{info.loc}}`;
    return `<span class="cilia-badge" title="${{title.replace(/"/g, '&quot;')}}">cilia</span>`;
}}

function getActiveGeneSet() {{
    if (geneFilterMode === 'all') return null;
    return new Set(CILIARY_SETS[geneFilterMode] || []);
}}

// ---- View Switching ----
function switchView(mode) {{
    currentMode = mode;
    const btnWhole = document.getElementById('btn-view-whole');
    const btnCy = document.getElementById('btn-view-cy');
    const btnClusters = document.getElementById('btn-all-clusters-head');
    const hud = document.getElementById('webgl-hud');
    const cyEl = document.getElementById('cy');
    const sigmaEl = document.getElementById('sigma-container');

    if (mode === 'whole') {{
        if (btnWhole) btnWhole.classList.add('active');
        if (btnCy) btnCy.classList.remove('active');
        if (btnClusters) btnClusters.classList.remove('active');
        if (hud) hud.style.display = 'block';
        if (sigmaEl) sigmaEl.style.display = 'block';
        if (cyEl) cyEl.style.display = 'none';
        if (sigmaRenderer) sigmaRenderer.refresh();
        updateStatus();
    }} else if (mode === 'gene') {{
        if (btnWhole) btnWhole.classList.remove('active');
        if (btnCy) btnCy.classList.add('active');
        if (btnClusters) btnClusters.classList.remove('active');
        if (hud) hud.style.display = 'none';
        if (sigmaEl) sigmaEl.style.display = 'none';
        if (cyEl) cyEl.style.display = 'block';
        if (cy) cy.resize();
        if (selectedGene) renderEgoNetwork(selectedGene);
        updateStatus();
    }} else if (mode === 'all_clusters') {{
        if (btnWhole) btnWhole.classList.remove('active');
        if (btnCy) btnCy.classList.remove('active');
        if (btnClusters) btnClusters.classList.add('active');
        if (hud) hud.style.display = 'none';
        if (sigmaEl) sigmaEl.style.display = 'none';
        if (cyEl) cyEl.style.display = 'block';
        if (cy) cy.resize();
        updateStatus();
    }}
}}

// ---- Binary Loaders ----
async function loadWholeGraph() {{
    try {{
        const resp = await fetch('network_whole.bin');
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        const buf = await resp.arrayBuffer();

        const header = new DataView(buf, 0, 12);
        const magic = String.fromCharCode(header.getUint8(0), header.getUint8(1), header.getUint8(2), header.getUint8(3));
        if (magic !== 'DLW1') throw new Error('Invalid whole graph magic: ' + magic);

        const nNodes = header.getUint32(4, true);
        const nEdges = header.getUint32(8, true);

        const nodeBlockOffset = 12;
        const edgeBlockOffset = 12 + nNodes * 8;

        const nodeView = new DataView(buf, nodeBlockOffset, nNodes * 8);
        const edgeView = new DataView(buf, edgeBlockOffset, nEdges * 6);

        // Build Graphology graph
        sigmaGraph = new graphology.Graph();
        for (let i = 0; i < nNodes; i++) {{
            const name = NAMES[i];
            const rawX = nodeView.getInt16(i * 8, true);
            const rawY = nodeView.getInt16(i * 8 + 2, true);
            const losses = nodeView.getUint16(i * 8 + 4, true);
            const cid = nodeView.getUint16(i * 8 + 6, true);
            GM[name] = losses;

            // Map [-2000, 2000] into normalized [0.05, 0.95] space for Sigma WebGL camera
            const x = ((rawX + 2000) / 4000) * 0.9 + 0.05;
            const y = ((rawY + 2000) / 4000) * 0.9 + 0.05;

            const color = getClusterColor(cid);
            const size = Math.max(3.5, Math.min(16, 3.5 + Math.sqrt(losses) * 0.75));

            sigmaGraph.addNode(name, {{
                x: x,
                y: y,
                size: size,
                label: name,
                color: color,
                baseColor: color,
                losses: losses,
                clusterId: cid
            }});
        }}

        for (let i = 0; i < nEdges; i++) {{
            const uIdx = edgeView.getUint16(i * 6, true);
            const vIdx = edgeView.getUint16(i * 6 + 2, true);
            const jRaw = edgeView.getUint16(i * 6 + 4, true);
            const u = NAMES[uIdx];
            const v = NAMES[vIdx];
            const j = jRaw / 65535.0;

            if (sigmaGraph.hasNode(u) && sigmaGraph.hasNode(v) && !sigmaGraph.hasEdge(u, v)) {{
                sigmaGraph.addEdge(u, v, {{
                    weight: j,
                    size: 0.5 + j * 2.0,
                    color: '#2a4365'
                }});
            }}
        }}

        // Compute centroids for clusters
        CLUSTER_CENTROIDS = {{}};
        for (const cid in CLUSTERS) {{
            let sx = 0, sy = 0, cnt = 0;
            for (const g of CLUSTERS[cid]) {{
                if (sigmaGraph.hasNode(g)) {{
                    const a = sigmaGraph.getNodeAttributes(g);
                    sx += a.x; sy += a.y; cnt++;
                }}
            }}
            if (cnt > 0) CLUSTER_CENTROIDS[cid] = {{ x: sx / cnt, y: sy / cnt, count: cnt }};
        }}

        if (!isWebGLAvailable()) {{
            console.warn('WebGL is unavailable in this browser environment. Defaulting to Cytoscape.');
            const hud = document.getElementById('webgl-hud');
            if (hud) hud.innerHTML = '<strong>Note:</strong> WebGL acceleration disabled. Use Gene Focus for full exploration.';
            switchView('gene');
            WHOLE_LOADED = true;
            return;
        }}

        // Initialize Sigma.js with crisp white labels and high contrast
        const container = document.getElementById('sigma-container');
        sigmaRenderer = new Sigma(sigmaGraph, container, {{
            renderEdgeLabels: false,
            enableEdgeClickEvents: false,
            enableEdgeWheelEvents: false,
            enableEdgeHoverEvents: false,
            labelColor: {{ color: '#ffffff' }},
            labelFont: 'Inter, system-ui, -apple-system, sans-serif',
            labelSize: 12,
            labelWeight: '600',
            labelRenderedSizeThreshold: 5,
            minCameraRatio: 0.02,
            maxCameraRatio: 8
        }});

        // Center camera precisely on the normalized [0, 1] graph
        sigmaRenderer.getCamera().setState({{ x: 0.5, y: 0.5, ratio: 1.05 }});

        // WebGL Interactions
        sigmaRenderer.on('enterNode', ({{ node }}) => {{
            const neighbors = new Set(sigmaGraph.neighbors(node));
            neighbors.add(node);

            sigmaRenderer.setSetting('nodeReducer', (n, data) => {{
                if (n === node) {{
                    return {{ ...data, zIndex: 30, color: '#ffffff', size: Math.max(data.size * 1.5, 12), label: data.label }};
                }}
                if (neighbors.has(n)) {{
                    return {{ ...data, zIndex: 20, color: '#38bdf8', size: Math.max(data.size, 8), label: data.label }};
                }}
                return {{ ...data, zIndex: 0, color: '#111827', label: '' }};
            }});

            sigmaRenderer.setSetting('edgeReducer', (e, data) => {{
                const [source, target] = sigmaGraph.extremities(e);
                if (source === node || target === node) {{
                    return {{ ...data, color: '#38bdf8', size: 2.0, zIndex: 5 }};
                }}
                return {{ ...data, color: '#0f172a', size: 0.1 }};
            }});

            const d = sigmaGraph.getNodeAttributes(node);
            const tooltip = document.getElementById('tooltip');
            tooltip.innerHTML = `<strong>${{d.label}}</strong>${{getCiliaBadgeHtml(d.label)}}<br>Losses: ${{d.losses}}<br><span style="color:${{d.baseColor}};">●</span> Cluster C${{d.clusterId}}: ${{CLUSTER_NAMES[d.clusterId] || ''}}`;
            tooltip.style.display = 'block';
        }});

        sigmaRenderer.on('leaveNode', () => {{
            sigmaRenderer.setSetting('nodeReducer', null);
            sigmaRenderer.setSetting('edgeReducer', null);
            document.getElementById('tooltip').style.display = 'none';
        }});

        sigmaRenderer.getMouseCaptor().on('mousemove', (e) => {{
            const tooltip = document.getElementById('tooltip');
            if (tooltip.style.display === 'block') {{
                tooltip.style.left = (e.clientX + 14) + 'px';
                tooltip.style.top = (e.clientY + 14) + 'px';
            }}
        }});

        sigmaRenderer.on('clickNode', ({{ node }}) => {{
            selectGene(node);
        }});

        WHOLE_LOADED = true;
        updateStatus();
    }} catch (err) {{
        console.error('Failed to load network_whole.bin:', err);
        document.getElementById('graph-status').textContent = 'Error loading whole graph: ' + err.message;
        switchView('gene');
    }}
}}

async function loadPartnersGraph() {{
    try {{
        const resp = await fetch('network_partners.bin');
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        const buf = await resp.arrayBuffer();

        const header = new DataView(buf, 0, 8);
        const magic = String.fromCharCode(header.getUint8(0), header.getUint8(1), header.getUint8(2), header.getUint8(3));
        if (magic !== 'DLP1') throw new Error('Invalid partners magic: ' + magic);

        const nGenes = header.getUint32(4, true);
        PARTNER_OFFSETS = new Uint32Array(buf, 8, nGenes + 1);
        PARTNER_LOSSES = new Uint16Array(buf, 8 + (nGenes + 1) * 4, nGenes);
        const partnerBlockOffset = 8 + (nGenes + 1) * 4 + nGenes * 2;
        PARTNER_DATA = new Uint16Array(buf, partnerBlockOffset);

        for (let i = 0; i < nGenes; i++) {{
            GM[NAMES[i]] = PARTNER_LOSSES[i];
        }}

        PARTNERS_LOADED = true;
        const pill = document.getElementById('loading-pill');
        if (pill) pill.style.display = 'none';
        updateStatus();
        if (selectedGene && currentMode === 'gene') {{
            renderEgoNetwork(selectedGene);
        }}
    }} catch (err) {{
        console.error('Failed to load network_partners.bin:', err);
    }}
}}

function getGeneData(name) {{
    if (G_CACHE[name]) return G_CACHE[name];
    if (!PARTNERS_LOADED || !PARTNER_DATA) return null;
    const idx = GENE_IDX[name];
    if (idx === undefined) return null;

    const byteOff = PARTNER_OFFSETS[idx];
    const u16Off = byteOff >> 1;
    const count = PARTNER_DATA[u16Off];
    const partners = [];
    let pos = u16Off + 1;
    for (let i = 0; i < count; i++) {{
        const pIdx = PARTNER_DATA[pos++];
        const jRaw = PARTNER_DATA[pos++];
        partners.push({{
            n: NAMES[pIdx],
            j: jRaw / 65535.0,
            l: GM[NAMES[pIdx]] || 0
        }});
    }}
    const res = {{ l: GM[name] || 0, p: partners }};
    G_CACHE[name] = res;
    return res;
}}

// ---- Selection & Navigation ----
function selectGene(name) {{
    if (!name || (GM[name] === undefined && GENE_IDX[name] === undefined)) return;
    selectedGene = name;
    document.getElementById('search').value = name;
    renderSidebar(name);

    if (currentMode === 'whole' && sigmaRenderer && sigmaGraph && sigmaGraph.hasNode(name)) {{
        const nodeAttrs = sigmaGraph.getNodeAttributes(name);
        sigmaRenderer.getCamera().animate({{ x: nodeAttrs.x, y: nodeAttrs.y, ratio: 0.15 }}, {{ duration: 500 }});
    }} else {{
        switchView('gene');
        renderEgoNetwork(name);
    }}
}}

function renderClusterDirectory() {{
    const sortedCids = Object.keys(CLUSTERS).map(Number).sort((a, b) => CLUSTERS[b].length - CLUSTERS[a].length);
    document.getElementById('gene-info').innerHTML = `
        <h2>🔬 Leiden Modules <span style="font-size:11px;color:#8892b0;font-weight:400;">(80 clusters)</span></h2>
        <div class="meta">Click a cluster to zoom in • Click "View Cluster →" to explore all members</div>
    `;

    let html = '';
    sortedCids.forEach(cid => {{
        const members = CLUSTERS[cid];
        const color = getClusterColor(cid);
        const cname = CLUSTER_NAMES[cid] || ('Cluster ' + cid);
        const cilCount = members.filter(m => ALL_CILIARY.has(m)).length;

        html += `
        <div class="cluster-entry" data-cluster-id="${{cid}}" onclick="zoomToCluster(${{cid}})" ondblclick="showSingleCluster(${{cid}})" title="C${{cid}}: ${{cname}}">
            <span class="swatch" style="background:${{color}};"></span>
            <div class="cname">
                <strong style="color:#7eb8ff;">C${{cid}}:</strong> ${{cname}}
            </div>
            ${{cilCount > 0 ? `<span class="cilia-badge">${{cilCount}} cil</span>` : ''}}
            <span class="csize">${{members.length}} genes</span>
            <button class="btn btn-accent" style="padding:2px 7px; font-size:10.5px; white-space:nowrap; margin-left:4px;" onclick="event.stopPropagation(); showSingleCluster(${{cid}});">View →</button>
        </div>`;
    }});
    document.getElementById('partner-list').innerHTML = html;
}}

function zoomToCluster(cid) {{
    if (currentMode === 'all_clusters' && cy) {{
        const cnodes = cy.nodes().filter(n => n.data('clusterId') === cid);
        if (cnodes.length > 0) {{
            cy.animate({{ fit: {{ eles: cnodes, padding: 60 }} }}, {{ duration: 400 }});
        }}
    }} else if (currentMode === 'whole' && sigmaRenderer && CLUSTER_CENTROIDS[cid]) {{
        sigmaRenderer.getCamera().animate({{
            x: CLUSTER_CENTROIDS[cid].x,
            y: CLUSTER_CENTROIDS[cid].y,
            ratio: 0.25
        }}, {{ duration: 500 }});
    }} else {{
        showSingleCluster(cid);
    }}
}}

function renderSidebar(name) {{
    const info = getGeneData(name) || {{ l: GM[name] || 0, p: [] }};
    const cid = GENE_CL[name];
    const cname = cid !== undefined ? (CLUSTER_NAMES[cid] || `Cluster ${{cid}}`) : null;
    const color = getClusterColor(cid);

    let clusterBadge = '';
    if (cid !== undefined) {{
        clusterBadge = `
        <div style="margin-top:8px; padding:6px 10px; background:rgba(26,32,64,0.85); border:1px solid #2a3558; border-radius:6px; display:flex; align-items:center; justify-content:space-between; gap:8px;">
            <div style="font-size:11.5px; color:#cbd5e1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
                <span style="display:inline-block; width:9px; height:9px; border-radius:2px; background:${{color}}; margin-right:5px;"></span>
                <strong style="color:#7eb8ff;">C${{cid}}:</strong> ${{cname}}
            </div>
            <button class="btn btn-accent" style="padding:2px 8px; font-size:11px; white-space:nowrap;" onclick="showSingleCluster(${{cid}}, '${{name}}')">View Cluster →</button>
        </div>`;
    }}

    const ciliaBadge = getCiliaBadgeHtml(name);
    const uniprotUrl = getUniProtUrl(name);

    document.getElementById('gene-info').innerHTML = `
        <a class="back-to-clusters" onclick="renderClusterDirectory();">← All Leiden Clusters</a>
        <h2>
            <span style="overflow:hidden; text-overflow:ellipsis;">${{name}} ${{ciliaBadge}}</span>
            <a href="${{uniprotUrl}}" target="_blank" rel="noopener noreferrer" class="gene-ext-link" title="Open in UniProt">UniProt ↗</a>
        </h2>
        <div class="meta">
            Independent loss events: <strong>${{info.l}}</strong>
        </div>
        ${{clusterBadge}}
    `;

    const thresh = parseFloat(document.getElementById('thresh').value);
    let topn = parseInt(document.getElementById('topn').value);
    if (document.getElementById('toggle-topn-max').checked) topn = Infinity;
    const activeSet = getActiveGeneSet();

    const qualifying = info.p.filter(p => p.j >= thresh && (!activeSet || activeSet.has(p.n))).slice(0, topn);

    let html = '';
    qualifying.forEach((p, idx) => {{
        const isCil = ALL_CILIARY.has(p.n);
        const pLoss = GM[p.n] !== undefined ? GM[p.n] : (p.l || 0);
        html += `
        <div class="partner" onclick="selectGene('${{p.n}}')">
            <span class="rank">#${{idx + 1}}</span>
            <span class="pname in-graph">${{p.n}}</span>
            ${{isCil ? getCiliaBadgeHtml(p.n) : ''}}
            <div class="bar-wrap"><div class="bar" style="width: ${{p.j * 100}}%;"></div></div>
            <span class="pjaccard">${{p.j.toFixed(2)}}</span>
            <span class="ploss">${{pLoss}}L</span>
            <a href="${{getUniProtUrl(p.n)}}" target="_blank" rel="noopener noreferrer" class="partner-ext-link" onclick="event.stopPropagation();" title="UniProt">↗</a>
        </div>`;
    }});

    if (qualifying.length === 0) {{
        html = '<div style="padding:16px; color:#556; text-align:center;">No partners meeting current filter criteria.</div>';
    }}

    document.getElementById('partner-list').innerHTML = html;
}}

// ---- Cytoscape Graph Rendering ----
function initCy() {{
    cy = cytoscape({{
        container: document.getElementById('cy'),
        style: [
            {{
                selector: 'node',
                style: {{
                    'label': 'data(label)',
                    'background-color': 'data(color)',
                    'width': 'data(size)',
                    'height': 'data(size)',
                    'color': '#f8fafc',
                    'font-size': 10,
                    'font-weight': 600,
                    'text-valign': 'center',
                    'text-halign': 'right',
                    'text-margin-x': 4,
                    'text-outline-color': '#0a0e17',
                    'text-outline-width': 2,
                    'min-zoomed-font-size': 5
                }}
            }},
            {{
                selector: 'node.focus',
                style: {{
                    'border-color': '#38bdf8',
                    'border-width': 3,
                    'font-size': 13,
                    'font-weight': 700,
                    'color': '#ffffff',
                    'z-index': 10
                }}
            }},
            {{
                selector: 'node.cluster-label',
                style: {{
                    'background-opacity': 0,
                    'border-width': 0,
                    'font-size': 13,
                    'font-weight': 'bold',
                    'color': '#ffffff',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'text-wrap': 'wrap',
                    'text-max-width': '160px',
                    'text-outline-color': '#0a0e17',
                    'text-outline-width': 2,
                    'min-zoomed-font-size': 4,
                    'z-index': 20
                }}
            }},
            {{
                selector: 'node.hide-label',
                style: {{
                    'text-opacity': 0
                }}
            }},
            {{
                selector: 'edge',
                style: {{
                    'width': 'data(width)',
                    'line-color': 'data(color)',
                    'opacity': 0.35,
                    'curve-style': 'haystack'
                }}
            }}
        ],
        layout: {{ name: 'preset' }},
        textureOnViewport: true,
        hideEdgesOnViewport: true,
        pixelRatio: 1.0
    }});

    cy.on('tap', 'node', function(evt) {{
        const d = evt.target.data();
        if (d.isLabel && d.clusterId !== undefined) {{
            zoomToCluster(d.clusterId);
        }} else if (d.id) {{
            selectGene(d.id);
        }}
    }});

    cy.on('dbltap', 'node', function(evt) {{
        const d = evt.target.data();
        if (d.clusterId !== undefined) {{
            showSingleCluster(d.clusterId);
        }}
    }});
}}

function renderEgoNetwork(geneName) {{
    if (!cy || !geneName) return;
    const info = getGeneData(geneName);
    if (!info) return;

    const thresh = parseFloat(document.getElementById('thresh').value);
    let topn = parseInt(document.getElementById('topn').value);
    if (document.getElementById('toggle-topn-max').checked) topn = Infinity;
    const activeSet = getActiveGeneSet();

    const qualifying = info.p.filter(p => p.j >= thresh && (!activeSet || activeSet.has(p.n))).slice(0, topn);
    const displayedSet = new Set(qualifying.map(p => p.n));
    displayedSet.add(geneName);

    const elements = [];
    // Focus node
    elements.push({{
        group: 'nodes',
        data: {{
            id: geneName,
            label: geneName,
            color: getClusterColor(GENE_CL[geneName]),
            size: 20
        }},
        classes: 'focus'
    }});

    // Partner nodes
    qualifying.forEach(p => {{
        elements.push({{
            group: 'nodes',
            data: {{
                id: p.n,
                label: p.n,
                color: getClusterColor(GENE_CL[p.n]),
                size: Math.max(10, Math.min(22, 10 + Math.sqrt(p.l) * 0.8))
            }}
        }});

        // Primary edges
        elements.push({{
            group: 'edges',
            data: {{
                id: `${{geneName}}--${{p.n}}`,
                source: geneName,
                target: p.n,
                width: 0.8 + p.j * 2.5,
                color: '#38bdf8'
            }}
        }});
    }});

    // Intra-partner edges
    qualifying.forEach(p => {{
        const pData = getGeneData(p.n);
        if (!pData) return;
        for (const pp of pData.p) {{
            if (pp.j >= thresh && displayedSet.has(pp.n) && p.n < pp.n && pp.n !== geneName) {{
                elements.push({{
                    group: 'edges',
                    data: {{
                        id: `${{p.n}}--${{pp.n}}`,
                        source: p.n,
                        target: pp.n,
                        width: 0.4 + pp.j * 1.8,
                        color: 'rgba(90, 142, 255, 0.25)'
                    }}
                }});
            }}
        }}
    }});

    cy.batch(() => {{
        cy.elements().remove();
        cy.add(elements);
        cy.layout({{
            name: 'concentric',
            concentric: function(node) {{ return node.id() === geneName ? 2 : 1; }},
            levelWidth: function() {{ return 1; }},
            padding: 50
        }}).run();
    }});

    updateStatus();
}}

function showSingleCluster(cid, highlightGene) {{
    if (!CLUSTERS[cid]) return;
    switchView('gene');
    currentClusterId = cid;

    const members = CLUSTERS[cid];
    const color = getClusterColor(cid);
    const cname = CLUSTER_NAMES[cid] || `Cluster ${{cid}}`;
    const thresh = parseFloat(document.getElementById('thresh').value);
    const topn = parseInt(document.getElementById('topn').value);

    const activeSet = getActiveGeneSet();
    const qualifying = members.filter(n => GM[n] !== undefined && (!activeSet || activeSet.has(n)));
    const shown = qualifying.slice(0, topn * 2);
    const shownSet = new Set(shown);

    document.getElementById('gene-info').innerHTML = `
        <a class="back-to-clusters" onclick="showAllClusters();">← All Leiden Clusters</a>
        <h2 style="color:${{color}};">C${{cid}}: ${{cname}}</h2>
        <div class="meta">${{shown.length}} of ${{members.length}} cluster genes shown</div>
    `;

    const elements = [];
    shown.forEach(name => {{
        elements.push({{
            group: 'nodes',
            data: {{
                id: name,
                label: name,
                color: color,
                size: Math.max(8, Math.min(20, 8 + Math.sqrt(GM[name] || 0) * 0.8)),
                clusterId: cid
            }},
            classes: name === highlightGene ? 'focus' : ''
        }});
    }});

    // Add intra-cluster edges
    shown.forEach(name => {{
        const d = getGeneData(name);
        if (!d) return;
        for (const p of d.p) {{
            if (p.j >= thresh && shownSet.has(p.n) && name < p.n) {{
                elements.push({{
                    group: 'edges',
                    data: {{
                        id: `${{name}}--${{p.n}}`,
                        source: name,
                        target: p.n,
                        width: 0.4 + p.j * 2,
                        color: color
                    }}
                }});
            }}
        }}
    }});

    cy.batch(() => {{
        cy.elements().remove();
        cy.add(elements);
        cy.layout({{ name: 'circle', padding: 40 }}).run();
    }});

    // Populate sidebar with cluster members
    let sideHtml = '';
    shown.forEach((m, i) => {{
        const isCil = ALL_CILIARY.has(m);
        sideHtml += `
        <div class="partner" onclick="selectGene('${{m}}')">
            <span class="rank">#${{i + 1}}</span>
            <span class="pname in-graph">${{m}}</span>
            ${{isCil ? getCiliaBadgeHtml(m) : ''}}
            <span class="ploss">${{GM[m] || 0}}L</span>
            <a href="${{getUniProtUrl(m)}}" target="_blank" rel="noopener noreferrer" class="partner-ext-link" onclick="event.stopPropagation();">↗</a>
        </div>`;
    }});
    document.getElementById('partner-list').innerHTML = sideHtml;
    updateStatus();
}}

function showAllClusters() {{
    switchView('all_clusters');
    selectedGene = null;
    document.getElementById('search').value = '';

    const thresh = parseFloat(document.getElementById('thresh').value);
    const MAX_NODES = 20;
    const sortedCids = Object.keys(CLUSTERS).map(Number).sort((a, b) => CLUSTERS[b].length - CLUSTERS[a].length);
    const cols = 9;
    const spacing = 520;

    const elements = [];

    sortedCids.forEach((cid, idx) => {{
        const members = CLUSTERS[cid];
        const color = getClusterColor(cid);
        const cname = CLUSTER_NAMES[cid] || ('Cluster ' + cid);
        const cx = (idx % cols) * spacing;
        const cy_pos = Math.floor(idx / cols) * spacing;

        const sorted = members.filter(n => GM[n] !== undefined).map(n => ({{ n, l: GM[n] }})).sort((a, b) => b.l - a.l);
        const vis = sorted.slice(0, MAX_NODES);
        const nVis = vis.length;
        const radius = Math.max(45, Math.min(150, Math.sqrt(nVis) * 28));

        vis.forEach((m, mi) => {{
            const angle = (2 * Math.PI * mi) / nVis;
            elements.push({{
                group: 'nodes',
                data: {{
                    id: m.n,
                    label: m.n.length > 12 ? m.n.slice(0, 10) + '…' : m.n,
                    fullName: m.n,
                    losses: m.l,
                    size: Math.max(6, Math.min(18, 6 + Math.sqrt(m.l) * 0.7)),
                    color: color,
                    clusterId: cid
                }},
                position: {{ x: cx + radius * Math.cos(angle), y: cy_pos + radius * Math.sin(angle) }}
            }});
        }});

        const shortName = cname.length > 22 ? cname.slice(0, 20) + '…' : cname;
        elements.push({{
            group: 'nodes',
            data: {{
                id: '__clabel_' + cid,
                label: 'C' + cid + ': ' + shortName + String.fromCharCode(10) + '(' + members.length + ' genes)',
                size: 1,
                color: 'transparent',
                clusterId: cid,
                isLabel: true
            }},
            position: {{ x: cx, y: cy_pos }},
            classes: 'cluster-label'
        }});
    }});

    cy.batch(() => {{
        cy.elements().remove();
        cy.add(elements);
    }});
    cy.fit(undefined, 40);

    renderClusterDirectory();
    updateStatus();
}}

function updateStatus() {{
    const status = document.getElementById('graph-status');
    if (!status) return;
    if (currentMode === 'whole') {{
        status.textContent = 'Mode: Whole Network (WebGL) • 11,236 genes • 73,467 edges';
    }} else if (currentMode === 'all_clusters') {{
        status.textContent = 'Mode: Leiden Clusters (Cytoscape) • 80 clusters • ' + cy.nodes().length + ' nodes';
    }} else if (cy) {{
        status.textContent = `Mode: Cytoscape Focus • ${{cy.nodes().length}} nodes • ${{cy.edges().length}} edges`;
    }}
}}

// Autocomplete search
const searchInput = document.getElementById('search');
const sugBox = document.getElementById('suggestions');

searchInput.addEventListener('input', function() {{
    const q = this.value.trim().toUpperCase();
    if (!q) {{ sugBox.style.display = 'none'; return; }}
    const matches = NAMES.filter(n => n.toUpperCase().includes(q)).slice(0, 15);
    if (matches.length === 0) {{ sugBox.style.display = 'none'; return; }}
    sugBox.innerHTML = matches.map(m => `<div onclick="selectGene('${{m}}'); document.getElementById('suggestions').style.display='none';">${{m}} ${{getCiliaBadgeHtml(m)}}</div>`).join('');
    sugBox.style.display = 'block';
}});

searchInput.addEventListener('keydown', function(e) {{
    if (e.key === 'Enter') {{
        const q = this.value.trim().toUpperCase();
        const exact = NAMES.find(n => n.toUpperCase() === q);
        if (exact) {{
            selectGene(exact);
            sugBox.style.display = 'none';
        }}
    }}
}});

document.addEventListener('click', (e) => {{
    if (!e.target.closest('.search-box')) sugBox.style.display = 'none';
}});

// Sidebar search & filter
(function() {{
    const sSearch = document.getElementById('sidebar-search');
    const ssClear = document.getElementById('sidebar-search-clear');

    function applySidebarFilter() {{
        const q = sSearch.value.trim().toLowerCase();
        ssClear.style.display = q ? 'block' : 'none';

        // Partner rows
        const partnerRows = document.querySelectorAll('#partner-list .partner');
        if (partnerRows.length) {{
            partnerRows.forEach(row => {{
                const text = row.textContent.toLowerCase();
                row.style.display = (!q || text.includes(q)) ? '' : 'none';
            }});
        }}

        // Cluster rows
        const clusterRows = document.querySelectorAll('#partner-list .cluster-entry');
        if (clusterRows.length) {{
            clusterRows.forEach(row => {{
                const text = row.textContent.toLowerCase();
                row.style.display = (!q || text.includes(q)) ? '' : 'none';
            }});
        }}
    }}

    sSearch.addEventListener('input', applySidebarFilter);
    ssClear.addEventListener('click', function() {{
        sSearch.value = '';
        applySidebarFilter();
        sSearch.focus();
    }});
}})();

// Labels toggle
document.getElementById('toggle-labels').addEventListener('change', function() {{
    const show = this.checked;
    if (sigmaRenderer) {{
        sigmaRenderer.setSetting('renderLabels', show);
    }}
    if (cy) {{
        cy.batch(() => {{
            if (show) cy.nodes().removeClass('hide-label');
            else cy.nodes().addClass('hide-label');
        }});
    }}
}});

// Sliders and filters
document.getElementById('thresh').addEventListener('input', function() {{
    document.getElementById('thresh-val').textContent = parseFloat(this.value).toFixed(2);
    if (selectedGene) renderSidebar(selectedGene);
    if (currentMode === 'gene' && selectedGene) renderEgoNetwork(selectedGene);
}});

document.getElementById('topn').addEventListener('input', function() {{
    document.getElementById('topn-val').textContent = this.value;
    if (selectedGene) renderSidebar(selectedGene);
    if (currentMode === 'gene' && selectedGene) renderEgoNetwork(selectedGene);
}});

document.getElementById('toggle-topn-max').addEventListener('change', function() {{
    document.getElementById('topn').disabled = this.checked;
    document.getElementById('topn-val').textContent = this.checked ? 'Max' : document.getElementById('topn').value;
    if (selectedGene) renderSidebar(selectedGene);
    if (currentMode === 'gene' && selectedGene) renderEgoNetwork(selectedGene);
}});

document.getElementById('gene-filter').addEventListener('change', function() {{
    geneFilterMode = this.value;
    if (selectedGene) renderSidebar(selectedGene);
    if (currentMode === 'gene' && selectedGene) renderEgoNetwork(selectedGene);
}});

// Init
window.addEventListener('DOMContentLoaded', () => {{
    initCy();
    loadWholeGraph();
    loadPartnersGraph();
    renderClusterDirectory();
}});

</script>
</body>
</html>
"""
    return html_code

if __name__ == "__main__":
    main()
