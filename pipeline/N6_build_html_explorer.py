#!/usr/bin/env python3
"""
Build a self-contained HTML network explorer:
1. Cytoscape.js: Interactive network explorer for Leiden modules, ego-networks, and shortest paths.
2. Species Tree: Circular eukaryotic phylogenetic tree (196 species cladogram/phylogram) with gene presence tracks.
3. Binary Streaming: Uses network_partners.bin (4.2 MB) for instant zero-lag loading.

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
    parser.add_argument("--top-k", type=int, default=500,
                        help="Max partners to store per gene in network_partners.bin (default: 500)")
    parser.add_argument("--min-loss", type=int, default=5,
                        help="Minimum losses for a gene to be included (default: 5)")
    parser.add_argument("--jaccard-floor", type=float, default=0.08,
                        help="Minimum Jaccard to include as partner (default: 0.08)")
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

    # 2. Build or verify network_partners.bin (Partner lists for Ego/Pathways)
    partners_bin_path = website_dir / "network_partners.bin"
    if not partners_bin_path.exists():
        print("2. Generating network_partners.bin...")
        generate_partners_binary(partners_bin_path, J, all_names, name_to_id,
                                 name_to_orig_idx, idx_to_name, loss_counts,
                                 args.top_k, args.jaccard_floor)
    else:
        print(f"2. Using existing {partners_bin_path}")

    # 3. Generate index.html
    print("3. Generating index.html...")
    output_path = Path(args.output or str(website_dir / "index.html"))
    html = build_html(all_names, cluster_data, cluster_names, cluster_colors, ciliary_sets, cilia_info)

    with open(output_path, "w") as f:
        f.write(html)

    size_kb = output_path.stat().st_size / 1024
    print(f"Successfully generated {output_path} ({size_kb:.1f} KB)")


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
#cy {{
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    display: block;
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
.partner.active-partner {{
    background: rgba(56, 189, 248, 0.22) !important;
    border-left: 3px solid #38bdf8;
}}
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
/* Light Theme Overrides */
body.light-theme {{
    background: #f8fafc;
    color: #0f172a;
}}
body.light-theme #header {{
    background: #ffffff;
    border-bottom: 1px solid #cbd5e1;
}}
body.light-theme #header h1 {{
    color: #0f172a;
}}
body.light-theme .controls label {{
    color: #475569;
}}
body.light-theme .controls select {{
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    color: #0f172a;
}}
body.light-theme .view-btn {{
    background: #f1f5f9;
    border-color: #cbd5e1;
    color: #334155;
}}
body.light-theme .view-btn.active {{
    background: #0284c7;
    border-color: #0284c7;
    color: #ffffff;
}}
body.light-theme .btn-clusters-header {{
    background: #f1f5f9;
    border-color: #cbd5e1;
    color: #0284c7;
}}
body.light-theme .btn {{
    background: #f1f5f9;
    border-color: #cbd5e1;
    color: #334155;
}}
body.light-theme .btn:hover {{
    background: #e2e8f0;
    color: #0f172a;
}}
body.light-theme .btn-accent {{
    background: #e0f2fe;
    border-color: #0284c7;
    color: #0284c7;
}}
body.light-theme #graph-wrapper {{
    background: #f8fafc;
}}
body.light-theme #jaccard-scale-bar span {{
    color: #475569;
}}
body.light-theme #jaccard-scale-bar div {{
    border-color: rgba(0, 0, 0, 0.15);
}}
body.light-theme #cy-edge-legend {{
    background: rgba(255, 255, 255, 0.92) !important;
    border: 1px solid #cbd5e1 !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08) !important;
}}
body.light-theme #cy-edge-legend div:first-child {{
    color: #334155 !important;
}}
body.light-theme #cy-edge-legend span {{
    color: #475569 !important;
}}
body.light-theme #sidebar {{
    background: #ffffff;
    border-left: 1px solid #e2e8f0;
    color: #0f172a;
}}
body.light-theme #sidebar-title,
body.light-theme #sidebar-search-wrap {{
    background: #ffffff;
    border-bottom-color: #e2e8f0;
}}
body.light-theme #sidebar-search {{
    background: #f8fafc;
    border-color: #cbd5e1;
    color: #0f172a;
}}
body.light-theme .cluster-card {{
    background: #f8fafc;
    border-color: #e2e8f0;
}}
body.light-theme .cluster-card:hover {{
    background: #f1f5f9;
    border-color: #0284c7;
}}
body.light-theme #status-bar {{
    background: #ffffff;
    border-top: 1px solid #e2e8f0;
    color: #64748b;
}}
body.light-theme #search {{
    background: #f8fafc;
    border-color: #cbd5e1;
    color: #0f172a;
}}
body.light-theme #suggestions {{
    background: #ffffff;
    border-color: #cbd5e1;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}}
body.light-theme #suggestions div {{
    color: #0f172a;
}}
body.light-theme #suggestions div:hover {{
    background: #f1f5f9;
}}
body.light-theme #gene-info h2 {{
    color: #0369a1;
}}
body.light-theme #gene-info .meta {{
    color: #64748b;
}}
body.light-theme #cluster-gene-focus-banner {{
    background: #f0f9ff !important;
    border: 1px solid #0284c7 !important;
    color: #0f172a !important;
}}
body.light-theme .partner {{
    border-bottom: 1px solid #e2e8f0;
    color: #0f172a;
}}
body.light-theme .partner:hover {{
    background: #f1f5f9;
}}
body.light-theme .partner.active {{
    background: #e0f2fe;
}}
body.light-theme .partner.active-partner {{
    background: #e0f2fe !important;
    border-left: 3px solid #0284c7;
}}
body.light-theme .partner .pname.in-graph {{
    color: #0284c7;
}}
body.light-theme .partner .bar-wrap {{
    background: #e2e8f0;
}}
body.light-theme .partner .rank {{
    color: #94a3b8;
}}

#tree-panel {{
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    display: none;
    flex-direction: column;
    overflow: hidden;
    z-index: 10;
}}
#tree-header {{
    height: 44px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 12px;
    gap: 10px;
    z-index: 10;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    flex-shrink: 0;
}}
.partner-tree-btn {{
    background: transparent;
    border: none;
    cursor: pointer;
    font-size: 12px;
    padding: 2px 4px;
    border-radius: 3px;
    color: #38bdf8;
    opacity: 0.65;
    transition: all 0.15s;
    line-height: 1;
    margin-right: 2px;
}}
.partner-tree-btn:hover {{
    opacity: 1;
    background: rgba(56, 189, 248, 0.2);
    transform: scale(1.15);
}}
.btn-tree-add {{
    background: #162035;
    border: 1px solid #2d3e60;
    color: #7dd3fc;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    cursor: pointer;
    font-weight: 500;
    transition: all 0.15s;
    display: inline-flex;
    align-items: center;
    gap: 3px;
}}
.btn-tree-add:hover {{
    background: #223554;
    border-color: #38bdf8;
    color: #ffffff;
}}
body.light-theme .partner-tree-btn {{
    color: #0284c7;
}}
body.light-theme .btn-tree-add {{
    background: #f1f5f9;
    border-color: #cbd5e1;
    color: #0369a1;
}}
body.light-theme .btn-tree-add:hover {{
    background: #e2e8f0;
    border-color: #0284c7;
}}
#tree-toast {{
    position: fixed;
    bottom: 24px;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(15,23,42,0.96);
    border: 1px solid #38bdf8;
    border-radius: 20px;
    padding: 7px 18px;
    color: #f8fafc;
    font-size: 12px;
    font-weight: 600;
    z-index: 99999;
    box-shadow: 0 8px 24px rgba(0,0,0,0.6);
    pointer-events: none;
    transition: opacity 0.25s ease;
}}
body.light-theme #tree-toast {{
    background: rgba(255, 255, 255, 0.98) !important;
    border-color: #0284c7 !important;
    color: #0f172a !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.15) !important;
}}

/* Tree Panel Specific Themes */
#tree-panel.tree-theme-dark,
#tree-modal.tree-theme-dark {{
    background: #070b14 !important;
}}
#tree-panel.tree-theme-dark #tree-header,
#tree-modal.tree-theme-dark #tree-header {{
    background: #0d1322 !important;
    border-bottom: 1px solid #1e293b !important;
}}
#tree-panel.tree-theme-dark #tree-header span,
#tree-modal.tree-theme-dark #tree-header span {{
    color: #f8fafc !important;
}}
#tree-panel.tree-theme-dark #tree-header label,
#tree-modal.tree-theme-dark #tree-header label {{
    color: #94a3b8 !important;
}}
#tree-panel.tree-theme-dark #tree-header select,
#tree-panel.tree-theme-dark #tree-header input,
#tree-modal.tree-theme-dark #tree-header select,
#tree-modal.tree-theme-dark #tree-header input {{
    background: #161d31 !important;
    border-color: #334155 !important;
    color: #f8fafc !important;
}}
#tree-panel.tree-theme-dark #tree-header .btn,
#tree-modal.tree-theme-dark #tree-header .btn {{
    background: #161d31 !important;
    border-color: #334155 !important;
    color: #f8fafc !important;
}}
#tree-panel.tree-theme-dark #tree-header .btn:hover,
#tree-modal.tree-theme-dark #tree-header .btn:hover {{
    background: #1e293b !important;
}}
#tree-panel.tree-theme-dark #tree-viewport,
#tree-modal.tree-theme-dark #tree-viewport {{
    background: #0a0e17 !important;
}}
#tree-panel.tree-theme-dark #tree-tooltip,
#tree-modal.tree-theme-dark #tree-tooltip {{
    background: rgba(15,23,42,0.96) !important;
    border: 1px solid #38bdf8 !important;
    color: #ffffff !important;
    box-shadow: 0 6px 20px rgba(0,0,0,0.5) !important;
}}

#tree-panel.tree-theme-light,
#tree-modal.tree-theme-light {{
    background: #f1f5f9 !important;
}}
#tree-panel.tree-theme-light #tree-header,
#tree-modal.tree-theme-light #tree-header {{
    background: #ffffff !important;
    border-bottom: 1px solid #cbd5e1 !important;
}}
#tree-panel.tree-theme-light #tree-header span,
#tree-modal.tree-theme-light #tree-header span {{
    color: #0f172a !important;
}}
#tree-panel.tree-theme-light #tree-header label,
#tree-modal.tree-theme-light #tree-header label {{
    color: #475569 !important;
}}
#tree-panel.tree-theme-light #tree-header select,
#tree-panel.tree-theme-light #tree-header input,
#tree-modal.tree-theme-light #tree-header select,
#tree-modal.tree-theme-light #tree-header input {{
    background: #f8fafc !important;
    border-color: #cbd5e1 !important;
    color: #0f172a !important;
}}
#tree-panel.tree-theme-light #tree-header .btn,
#tree-modal.tree-theme-light #tree-header .btn {{
    background: #f1f5f9 !important;
    border-color: #cbd5e1 !important;
    color: #334155 !important;
}}
#tree-panel.tree-theme-light #tree-header .btn:hover,
#tree-modal.tree-theme-light #tree-header .btn:hover {{
    background: #e2e8f0 !important;
    color: #0f172a !important;
}}
#tree-panel.tree-theme-light #tree-viewport,
#tree-modal.tree-theme-light #tree-viewport {{
    background: #ffffff !important;
}}
#tree-panel.tree-theme-light #tree-tooltip,
#tree-modal.tree-theme-light #tree-tooltip {{
    background: rgba(255,255,255,0.98) !important;
    border: 1px solid #0284c7 !important;
    color: #0f172a !important;
    box-shadow: 0 6px 20px rgba(0,0,0,0.15) !important;
}}
</style>
</head>
<body>

<div id="header">
    <h1>Dollo Co-Loss Network Explorer</h1>
    <div class="view-toggle">
        <button id="btn-view-cy" class="view-btn active" onclick="switchView('network')">🌐 Network View</button>
        <button id="btn-all-clusters-head" class="view-btn" onclick="switchView('all_clusters')">🗂️ Leiden Modules</button>
        <button id="btn-view-tree" class="view-btn" onclick="switchView('tree')" title="Species tree and gene presence tracks">🌳 Species Tree</button>
    </div>

    <div class="search-box">
        <input type="text" id="search" placeholder="Search gene (e.g. SCAPER, CEP290)…" autocomplete="off">
        <div id="suggestions"></div>
    </div>

    <div class="controls">
        <label>Jaccard &ge; <span id="thresh-val">0.20</span></label>
        <input type="range" id="thresh" min="0.10" max="0.70" step="0.02" value="0.20">

        <div id="jaccard-scale-bar" title="Edge color scale by Jaccard similarity" style="display:flex; align-items:center; gap:4px; margin-left:2px; margin-right:4px;">
            <span style="font-size:9.5px; color:#8892b0; font-family:ui-monospace, monospace;">0.20</span>
            <div style="width:65px; height:8px; border-radius:4px; background:linear-gradient(to right, #38bdf8, #2dd4bf, #34d399, #fbbf24, #f43f5e); border:1px solid rgba(255,255,255,0.18);"></div>
            <span style="font-size:9.5px; color:#8892b0; font-family:ui-monospace, monospace;">&ge;0.65</span>
        </div>

        <label>Show Top <span id="topn-val">25</span></label>
        <input type="range" id="topn" min="5" max="300" step="5" value="25">
        <label><input type="checkbox" id="toggle-topn-max"> Max</label>

        <label style="margin-left:4px;">Layout:</label>
        <select id="layout-select" style="background:#0d1220; border:1px solid #3a4570; color:#e0e6f0; padding:3px 6px; border-radius:4px; font-size:11px;">
            <option value="cose" selected>Force (Spread)</option>
            <option value="concentric">Concentric (Radial)</option>
        </select>

        <label>Filter:</label>
        <select id="gene-filter">
            <option value="all">All Genes</option>
            <option value="all_ciliary">All Union (SCGSv2 ∪ CiliaCarta)</option>
            <option value="ciliacarta">CiliaCarta</option>
            <option value="syscilia_v2">SYSCILIA v2</option>
            <option value="shared_core">Merged / Core (SCGSv2 ∩ CiliaCarta)</option>
        </select>

        <label style="margin-left:4px;">Labels:</label>
        <input type="checkbox" id="toggle-labels" checked>
        <button id="btn-theme-toggle" class="btn" style="margin-left:6px;" onclick="toggleSiteTheme()" title="Toggle Dark/Light Site Theme">🌙 Dark</button>
    </div>
</div>

<div id="main">
    <div id="graph-wrapper">
        <div id="cy"></div>
        <div id="cy-edge-legend" style="position:absolute; bottom:16px; left:16px; background:rgba(13, 18, 32, 0.85); backdrop-filter:blur(6px); border:1px solid #2a3558; border-radius:6px; padding:6px 10px; z-index:5; display:flex; flex-direction:column; gap:4px; pointer-events:none; box-shadow:0 4px 12px rgba(0,0,0,0.4); transition:opacity 0.2s;">
            <div style="font-size:10px; font-weight:600; color:#cbd5e1; text-transform:uppercase; letter-spacing:0.5px; display:flex; align-items:center; gap:5px;">
                <span style="display:inline-block; width:6px; height:6px; border-radius:50%; background:#38bdf8;"></span>
                Co-Loss Strength (Jaccard)
            </div>
            <div style="display:flex; align-items:center; gap:6px;">
                <span style="font-size:9.5px; color:#94a3b8; font-family:ui-monospace, monospace;">0.20</span>
                <div style="width:110px; height:8px; border-radius:4px; background:linear-gradient(to right, #38bdf8, #2dd4bf, #34d399, #fbbf24, #f43f5e); border:1px solid rgba(255,255,255,0.2);"></div>
                <span style="font-size:9.5px; color:#94a3b8; font-family:ui-monospace, monospace;">&ge;0.65</span>
            </div>
        </div>
        <div id="tree-panel" class="tree-theme-dark" style="display:none; position:absolute; inset:0; flex-direction:column; overflow:hidden; z-index:10;">
            <div id="tree-header">
                <div style="display:flex; align-items:center; gap:8px; flex:1; min-width:0; overflow:hidden;">
                    <span style="font-size:13px; font-weight:700; color:#f8fafc; white-space:nowrap; display:flex; align-items:center; gap:4px; flex-shrink:0;">
                        🌳 Species Tree <span style="font-size:10.5px; color:#8892b0; font-weight:400;">(196)</span>
                    </span>
                    <div style="position:relative; width:150px; flex-shrink:0;">
                        <input type="text" id="tree-gene-input" placeholder="+ Add gene…" autocomplete="off" style="width:100%; height:26px; background:#161d31; border:1px solid #2d3748; border-radius:4px; color:#e2e8f0; padding:0 8px; font-size:11px; outline:none;">
                        <div id="tree-gene-suggestions" style="display:none; position:absolute; top:30px; left:0; right:0; background:#0f172a; border:1px solid #334155; border-radius:4px; max-height:220px; overflow-y:auto; z-index:100; box-shadow:0 8px 24px rgba(0,0,0,0.6);"></div>
                    </div>
                    <div id="tree-chips-container" style="display:flex; align-items:center; gap:5px; flex-wrap:nowrap; overflow-x:auto; min-width:0;"></div>
                </div>
                <div style="display:flex; align-items:center; gap:6px; flex-shrink:0;">
                    <label style="font-size:11px; color:#94a3b8; display:flex; align-items:center; gap:4px; font-weight:500;">
                        Taxonomy:
                        <select id="tree-tax-select" style="background:#161d31; border:1px solid #334155; color:#f8fafc; border-radius:4px; padding:3px 6px; font-size:11px; outline:none; cursor:pointer;" onchange="setTreeTaxLevel(this.value)">
                            <option value="kingdom" selected>Kingdom (UniProt)</option>
                            <option value="phylum">Phylum (UniProt)</option>
                            <option value="supergroup">Supergroup</option>
                            <option value="tcs">TCS Major Clades (31)</option>
                            <option value="detailed">Detailed (Phylum)</option>
                        </select>
                    </label>
                    <label style="font-size:11px; color:#94a3b8; display:flex; align-items:center; gap:4px; font-weight:500;">
                        Branches:
                        <select id="tree-branch-mode" style="background:#161d31; border:1px solid #334155; color:#f8fafc; border-radius:4px; padding:3px 6px; font-size:11px; outline:none; cursor:pointer;" onchange="setTreeBranchMode(this.value)">
                            <option value="genes">Gene Parsimony</option>
                            <option value="taxonomy">Taxonomy Clade</option>
                        </select>
                    </label>
                    <label style="font-size:11px; color:#94a3b8; display:flex; align-items:center; gap:4px; font-weight:500;">
                        Length:
                        <select id="tree-branch-len-select" style="background:#161d31; border:1px solid #334155; color:#f8fafc; border-radius:4px; padding:3px 6px; font-size:11px; outline:none; cursor:pointer;" onchange="setTreeBranchLength(this.value)" title="Ignore branch lengths (cladogram like in iTOL) or use molecular clock branch lengths">
                            <option value="cladogram" selected>Ignore (Cladogram)</option>
                            <option value="phylogram">Use Branch Lengths</option>
                        </select>
                    </label>
                    <label style="font-size:11px; color:#94a3b8; display:flex; align-items:center; gap:4px; font-weight:500;">
                        Theme:
                        <select id="tree-theme-select" style="background:#161d31; border:1px solid #334155; color:#f8fafc; border-radius:4px; padding:3px 6px; font-size:11px; outline:none; cursor:pointer;" onchange="setTreeTheme(this.value)" title="Choose tree appearance mode">
                            <option value="auto" selected>Auto (Site)</option>
                            <option value="dark">🌙 Dark</option>
                            <option value="light">☀️ Light</option>
                        </select>
                    </label>
                    <button class="btn" style="padding:4px 8px; font-size:11px;" onclick="resetTreeGenes([])" title="Clear all genes on tree">Clear Genes</button>
                    <button class="btn" style="padding:4px 8px; font-size:11px;" onclick="loadCurrentNetworkGenesInTree()" title="Load currently selected gene and top partners">+ Current Gene</button>
                    <button class="btn" style="padding:4px 8px; font-size:11px;" onclick="resetTreeZoom()" title="Reset Pan & Zoom">⟲ Reset</button>
                    <button class="btn" style="padding:4px 8px; font-size:11px;" onclick="exportTreeSvg()" title="Download high-res SVG">📥 SVG</button>
                </div>
            </div>
            <div id="tree-viewport" style="flex:1; position:relative; overflow:hidden; background:#0a0e17; cursor:grab; user-select:none;">
                <div id="tree-loading" style="position:absolute; inset:0; display:flex; align-items:center; justify-content:center; background:rgba(10,14,23,0.85); color:#cbd5e1; font-size:14px; font-weight:600; z-index:50;">
                    <div class="spinner" style="margin-right:10px;"></div> Loading species tree and presence matrix…
                </div>
                <div id="tree-tooltip" style="position:absolute; display:none; pointer-events:none; background:rgba(15,23,42,0.96); border:1px solid #38bdf8; border-radius:6px; padding:8px 12px; color:#ffffff; font-size:11.5px; z-index:100; box-shadow:0 6px 20px rgba(0,0,0,0.5); max-width:320px; line-height:1.4;"></div>
                <div id="tree-canvas-wrapper" style="width:100%; height:100%; display:flex; align-items:center; justify-content:center; transform-origin:center center;">
                    <svg id="tree-svg" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000" width="1000" height="1000" style="max-width:100%; max-height:100%; font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;"></svg>
                </div>
                <!-- Zoom HUD -->
                <div style="position:absolute; right:16px; bottom:16px; display:flex; flex-direction:column; gap:4px; z-index:10;">
                    <button class="btn" style="width:30px; height:30px; padding:0; font-size:16px; font-weight:700; background:#0f172a; border:1px solid #334155; color:#fff;" onclick="zoomTree(1.25)">+</button>
                    <button class="btn" style="width:30px; height:30px; padding:0; font-size:16px; font-weight:700; background:#0f172a; border:1px solid #334155; color:#fff;" onclick="zoomTree(0.8)">−</button>
                    <button class="btn" style="width:30px; height:30px; padding:0; font-size:11px; background:#0f172a; border:1px solid #334155; color:#fff;" onclick="resetTreeZoom()">Fit</button>
                </div>
            </div>
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
        <div id="cluster-gene-focus-banner" style="display:none; margin:0 12px 10px 12px; padding:8px 10px; background:rgba(26, 38, 66, 0.95); border:1px solid #38bdf8; border-radius:6px;"></div>
        <div id="partner-list"></div>
    </div>
</div>

<div id="tree-toast" style="display:none;"></div>

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

// Cluster color helper (returns Hex color for cluster ID)
function getClusterColor(cid) {{
    if (cid !== undefined && CLUSTER_COLORS[cid]) return CLUSTER_COLORS[cid];
    return '#38bdf8';
}}

// ---- Global State ----
let currentMode = 'gene'; // 'gene' | 'all_clusters' | 'single_cluster'
let selectedGene = null;
let currentClusterId = null;
let clusterFocusedGene = null;
let geneFilterMode = 'all';

// ---- Theme Management (Site & Tree) ----
let SITE_THEME = localStorage.getItem('dollo_site_theme') || 'dark';
let TREE_THEME = localStorage.getItem('dollo_tree_theme') || 'auto';

function getEffectiveTreeTheme() {{
    if (TREE_THEME === 'auto') {{
        return SITE_THEME || 'dark';
    }}
    return TREE_THEME;
}}

function updateTreeModalTheme() {{
    const modal = document.getElementById('tree-panel') || document.getElementById('tree-modal');
    if (!modal) return;
    const effTheme = getEffectiveTreeTheme();
    if (effTheme === 'light') {{
        modal.classList.add('tree-theme-light');
        modal.classList.remove('tree-theme-dark');
    }} else {{
        modal.classList.add('tree-theme-dark');
        modal.classList.remove('tree-theme-light');
    }}
    const sel = document.getElementById('tree-theme-select');
    if (sel && sel.value !== TREE_THEME) {{
        sel.value = TREE_THEME;
    }}
    if (modal.style.display !== 'none' && TREE_LAYOUT) {{
        renderTreeChips();
        renderCircularTree();
    }}
}}

function showTreeToast(msg) {{
    const toast = document.getElementById('tree-toast');
    if (!toast) return;
    toast.textContent = msg;
    toast.style.display = 'block';
    toast.style.opacity = '1';
    clearTimeout(toast._timer);
    toast._timer = setTimeout(() => {{
        toast.style.opacity = '0';
        setTimeout(() => {{ toast.style.display = 'none'; }}, 260);
    }}, 2400);
}}

function setTreeTheme(theme) {{
    TREE_THEME = theme;
    localStorage.setItem('dollo_tree_theme', theme);
    updateTreeModalTheme();
}}

function applySiteTheme(theme) {{
    SITE_THEME = theme;
    localStorage.setItem('dollo_site_theme', theme);
    const btn = document.getElementById('btn-theme-toggle');
    if (theme === 'light') {{
        document.body.classList.add('light-theme');
        if (btn) btn.innerHTML = '☀️ Light';
    }} else {{
        document.body.classList.remove('light-theme');
        if (btn) btn.innerHTML = '🌙 Dark';
    }}
    updateTreeModalTheme();
    updateCyTheme();
}}

var cy = null;

function updateCyTheme() {{
    if (!cy) return;
    const isDark = (SITE_THEME !== 'light');
    const textBg = isDark ? '#090d16' : '#ffffff';
    const textCol = isDark ? '#f8fafc' : '#0f172a';
    const textBorder = isDark ? '#2a3550' : '#cbd5e1';
    cy.style()
        .selector('node')
        .style({{
            'color': textCol,
            'text-background-color': textBg,
            'text-border-color': textBorder
        }})
        .update();
}}

function toggleSiteTheme() {{
    const newTheme = (SITE_THEME === 'dark') ? 'light' : 'dark';
    applySiteTheme(newTheme);
}}

// Initialize site theme immediately
applySiteTheme(SITE_THEME);

// Binary buffers
let PARTNERS_LOADED = false;
let GM = {{}}; // loss counts
let PARTNER_OFFSETS = null;
let PARTNER_LOSSES = null;
let PARTNER_DATA = null;
const G_CACHE = {{}};
let CLUSTER_CENTROIDS = {{}};

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
    if (info && info.loc) title += `\\nLocalization: ${{info.loc}}`;
    return `<span class="cilia-badge" title="${{title.replace(/"/g, '&quot;')}}">cilia</span>`;
}}

function getActiveGeneSet() {{
    if (geneFilterMode === 'all') return null;
    return new Set(CILIARY_SETS[geneFilterMode] || []);
}}

// ---- View Switching ----
function updateControlsForMode(isLeiden) {{
    // Grey out Jaccard & Top-N controls when in Leiden cluster or Tree mode (they don't apply)
    const ids = ['thresh', 'topn', 'toggle-topn-max'];
    ids.forEach(id => {{
        const el = document.getElementById(id);
        if (el) el.disabled = isLeiden;
    }});
    const labels = document.querySelectorAll('.controls label');
    labels.forEach(lbl => {{
        const text = lbl.textContent.trim();
        if (text.startsWith('Jaccard') || text.startsWith('Show Top') || text.startsWith('Max')) {{
            lbl.style.opacity = isLeiden ? '0.35' : '1';
        }}
    }});
    const scaleBar = document.getElementById('jaccard-scale-bar');
    if (scaleBar) scaleBar.style.opacity = (currentMode === 'tree') ? '0.35' : '1';
    const cyLegend = document.getElementById('cy-edge-legend');
    if (cyLegend) cyLegend.style.display = (currentMode === 'tree') ? 'none' : 'flex';
}}

function switchView(mode) {{
    currentMode = mode;
    const btnNet = document.getElementById('btn-view-cy');
    const btnClusters = document.getElementById('btn-all-clusters-head');
    const btnTree = document.getElementById('btn-view-tree');
    const cyEl = document.getElementById('cy');
    const treeEl = document.getElementById('tree-panel');

    const isLeiden = (mode === 'single_cluster' || mode === 'all_clusters');
    updateControlsForMode(isLeiden || mode === 'tree');

    if (btnNet) btnNet.classList.toggle('active', mode === 'gene' || mode === 'network');
    if (btnClusters) btnClusters.classList.toggle('active', mode === 'all_clusters' || mode === 'single_cluster');
    if (btnTree) btnTree.classList.toggle('active', mode === 'tree');

    if (mode === 'gene' || mode === 'network') {{
        if (treeEl) treeEl.style.display = 'none';
        if (cyEl) cyEl.style.display = 'block';
        if (cy) cy.resize();
        if (selectedGene) {{
            renderEgoNetwork(selectedGene);
        }} else {{
            showAllClusters();
        }}
        updateStatus();
    }} else if (mode === 'single_cluster') {{
        if (treeEl) treeEl.style.display = 'none';
        if (cyEl) cyEl.style.display = 'block';
        if (cy) cy.resize();
        updateStatus();
    }} else if (mode === 'all_clusters') {{
        if (treeEl) treeEl.style.display = 'none';
        if (cyEl) cyEl.style.display = 'block';
        if (cy) cy.resize();
        showAllClusters();
    }} else if (mode === 'tree') {{
        if (cyEl) cyEl.style.display = 'none';
        if (treeEl) treeEl.style.display = 'flex';
        initTreeView();
        updateTreeModalTheme();
        updateStatus();
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
        if (selectedGene && (currentMode === 'gene' || currentMode === 'network')) {{
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
    if (currentMode === 'tree') {{
        addTreeGene(name);
    }} else {{
        switchView('network');
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
    }} else {{
        showSingleCluster(cid);
    }}
}}

function getJaccardColor(j) {{
    if (j === undefined || j === null) return '#38bdf8';
    const stops = [
        {{ t: 0.00, r: 56,  g: 189, b: 248 }},
        {{ t: 0.25, r: 45,  g: 212, b: 191 }},
        {{ t: 0.50, r: 52,  g: 211, b: 153 }},
        {{ t: 0.75, r: 251, g: 191, b: 36  }},
        {{ t: 1.00, r: 244, g: 63,  b: 94  }}
    ];
    const t = Math.max(0, Math.min(1, (j - 0.20) / (0.65 - 0.20)));
    for (let i = 0; i < stops.length - 1; i++) {{
        const s1 = stops[i];
        const s2 = stops[i + 1];
        if (t >= s1.t && t <= s2.t) {{
            const f = (t - s1.t) / (s2.t - s1.t);
            const r = Math.round(s1.r + f * (s2.r - s1.r));
            const g = Math.round(s1.g + f * (s2.g - s1.g));
            const b = Math.round(s1.b + f * (s2.b - s1.b));
            return `rgb(${{r}}, ${{g}}, ${{b}})`;
        }}
    }}
    return '#f43f5e';
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
            <div style="display:flex; align-items:center; gap:5px; flex-shrink:0;">
                <button class="btn btn-tree-add" onclick="addTreeGene('${{name}}')" title="Add ${{name}} to Species Tree">🌳 Add to Tree</button>
                <a href="${{uniprotUrl}}" target="_blank" rel="noopener noreferrer" class="gene-ext-link" title="Open in UniProt">UniProt ↗</a>
            </div>
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
            <div class="bar-wrap"><div class="bar" style="width: ${{p.j * 100}}%; background: ${{getJaccardColor(p.j)}};"></div></div>
            <span class="pjaccard">${{p.j.toFixed(2)}}</span>
            <span class="ploss">${{pLoss}}L</span>
            <button class="partner-tree-btn" onclick="event.stopPropagation(); addTreeGene('${{p.n}}');" title="Add ${{p.n}} to Species Tree">🌳</button>
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
                    'text-valign': 'bottom',
                    'text-halign': 'center',
                    'text-margin-y': 4,
                    'text-background-color': '#090d16',
                    'text-background-opacity': 0.85,
                    'text-background-padding': '2.5px',
                    'text-background-shape': 'round-rectangle',
                    'text-border-color': '#2a3550',
                    'text-border-width': 0.5,
                    'text-border-opacity': 0.8,
                    'min-zoomed-font-size': 5
                }}
            }},
            {{
                selector: 'node.focus',
                style: {{
                    'border-color': '#38bdf8',
                    'border-width': 3.5,
                    'font-size': 13.5,
                    'font-weight': 900,
                    'color': '#ffffff',
                    'text-valign': 'top',
                    'text-margin-y': -6,
                    'text-background-color': '#0369a1',
                    'text-background-opacity': 0.95,
                    'text-border-color': '#38bdf8',
                    'text-border-width': 1.5,
                    'z-index': 25
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
                selector: 'node.highlighted',
                style: {{
                    'border-color': '#38bdf8',
                    'border-width': 2.5,
                    'z-index': 9
                }}
            }},
            {{
                selector: 'node.dimmed',
                style: {{
                    'opacity': 0.22
                }}
            }},
            {{
                selector: 'edge',
                style: {{
                    'width': 'data(width)',
                    'line-color': 'data(color)',
                    'opacity': 'data(opacity)',
                    'curve-style': 'haystack'
                }}
            }},
            {{
                selector: 'edge.highlighted',
                style: {{
                    'line-color': 'data(color)',
                    'opacity': 1.0,
                    'width': 3.5,
                    'z-index': 8
                }}
            }},
            {{
                selector: 'edge.dimmed',
                style: {{
                    'opacity': 0.05
                }}
            }}
        ],
        layout: {{ name: 'preset' }},
        textureOnViewport: true,
        hideEdgesOnViewport: true,
        pixelRatio: 1.0
    }});
    window.cyInstance = cy;

    cy.on('tap', 'node', function(evt) {{
        const node = evt.target;
        const d = node.data();
        if (d.isLabel && d.clusterId !== undefined) {{
            zoomToCluster(d.clusterId);
            return;
        }}
        const geneId = d.id;
        if (!geneId) return;

        // Check if currently viewing a Leiden cluster
        if (currentMode === 'single_cluster' || currentMode === 'all_clusters') {{
            if (clusterFocusedGene === geneId) {{
                // SECOND CLICK: go to individual interactors!
                unfocusClusterGene();
                selectGene(geneId);
            }} else {{
                // FIRST CLICK: zoom in to gene in leiden cluster!
                focusGeneInCluster(node, geneId, d.clusterId || currentClusterId);
            }}
            return;
        }}

        // In individual ego network mode:
        selectGene(geneId);
    }});

    cy.on('dbltap', 'node', function(evt) {{
        const d = evt.target.data();
        if (d.isLabel && d.clusterId !== undefined) {{
            showSingleCluster(d.clusterId);
        }} else if (d.id) {{
            // Double tap directly navigates to individual interactors
            unfocusClusterGene();
            selectGene(d.id);
        }}
    }});

    cy.on('tap', function(evt) {{
        if (evt.target === cy) {{
            if ((currentMode === 'single_cluster' || currentMode === 'all_clusters') && clusterFocusedGene) {{
                unfocusClusterGene();
                if (currentMode === 'single_cluster') {{
                    cy.animate({{ fit: {{ padding: 60 }} }}, {{ duration: 350 }});
                }}
            }}
        }}
    }});

    // Hover tooltip for Cytoscape nodes
    const tipEl = document.getElementById('tooltip');
    cy.on('mouseover', 'node', function(evt) {{
        const d = evt.target.data();
        if (d.isLabel || !tipEl) return;
        const cid = d.clusterId !== undefined ? d.clusterId : GENE_CL[d.id];
        const cname = cid !== undefined ? (CLUSTER_NAMES[cid] || `Cluster ${{cid}}`) : '';
        const color = getClusterColor(cid);
        const losses = GM[d.id] !== undefined ? GM[d.id] : (d.losses || 0);
        let tipHtml = `<strong>${{d.id}}</strong>${{getCiliaBadgeHtml(d.id)}}<br>Losses: ${{losses}}`;
        if (cid !== undefined) {{
            tipHtml += `<br><span style="color:${{color}}; font-weight:700;">●</span> C${{cid}}: ${{cname}}`;
        }}
        tipEl.innerHTML = tipHtml;
        tipEl.style.display = 'block';
    }});
    cy.on('mouseout', 'node', function() {{
        if (tipEl) tipEl.style.display = 'none';
    }});
    cy.on('mouseover', 'edge', function(evt) {{
        const d = evt.target.data();
        if (!tipEl || d.jaccard === undefined) return;
        const jCol = getJaccardColor(d.jaccard);
        tipEl.innerHTML = `<strong>${{d.source}} ⟷ ${{d.target}}</strong><br><span style="display:inline-block; width:9px; height:9px; border-radius:50%; background:${{jCol}}; margin-right:5px; vertical-align:middle; border:1px solid rgba(255,255,255,0.4);"></span>Jaccard Similarity: <strong>${{d.jaccard.toFixed(2)}}</strong>`;
        tipEl.style.display = 'block';
    }});
    cy.on('mouseout', 'edge', function() {{
        if (tipEl) tipEl.style.display = 'none';
    }});
    cy.on('pan zoom', function() {{
        if (tipEl) tipEl.style.display = 'none';
    }});
    const cyContainer = document.getElementById('cy');
    if (cyContainer) {{
        cyContainer.addEventListener('mousemove', function(e) {{
            if (tipEl && tipEl.style.display === 'block') {{
                tipEl.style.left = (e.clientX + 14) + 'px';
                tipEl.style.top = (e.clientY + 14) + 'px';
            }}
        }});
    }}
}}

function focusGeneInCluster(node, geneId, clusterId) {{
    clusterFocusedGene = geneId;

    // Highlight node and connected cluster elements
    cy.elements().removeClass('focus highlighted dimmed');
    node.addClass('focus');

    const connEdges = node.connectedEdges();
    const neighbors = connEdges.connectedNodes();
    connEdges.addClass('highlighted');
    neighbors.addClass('highlighted');

    const others = cy.elements().difference(node.union(connEdges).union(neighbors));
    others.addClass('dimmed');

    // Zoom in smoothly to gene within cluster
    if (currentMode === 'single_cluster') {{
        const neighborhood = node.closedNeighborhood();
        if (neighborhood.nodes().length > 1) {{
            cy.animate({{
                fit: {{ eles: neighborhood, padding: 100 }}
            }}, {{ duration: 400 }});
        }} else {{
            cy.animate({{
                center: {{ eles: node }},
                zoom: Math.max(cy.zoom() * 1.5, 1.8)
            }}, {{ duration: 400 }});
        }}
    }} else if (currentMode === 'all_clusters') {{
        cy.animate({{
            center: {{ eles: node }},
            zoom: 2.2
        }}, {{ duration: 400 }});
    }}

    // Highlight row in sidebar and scroll into view
    document.querySelectorAll('.partner').forEach(el => {{
        if (el.getAttribute('data-gene') === geneId) {{
            el.classList.add('active-partner');
            el.scrollIntoView({{ block: 'nearest', behavior: 'smooth' }});
        }} else {{
            el.classList.remove('active-partner');
        }}
    }});

    // Banner prompt in sidebar
    const banner = document.getElementById('cluster-gene-focus-banner');
    if (banner) {{
        banner.style.display = 'block';
        banner.innerHTML = `
            <div style="display:flex; align-items:center; justify-content:space-between; gap:6px;">
                <div>
                    <strong style="color:#38bdf8; font-size:13px;">${{geneId}}</strong>
                    <span style="color:#94a3b8; font-size:11px; margin-left:4px;">${{GM[geneId] || 0}} losses</span>
                    ${{ALL_CILIARY.has(geneId) ? '<span class="cilia-badge" style="margin-left:4px;">CILIA</span>' : ''}}
                </div>
                <button class="btn btn-accent" style="padding:2px 8px; font-size:11px;" onclick="unfocusClusterGene(); selectGene('${{geneId}}');">
                    Open Interactors →
                </button>
            </div>
            <div style="font-size:10.5px; color:#94a3b8; margin-top:3px;">
                🔍 Focused in cluster • <strong>Click node again</strong> to view individual interactors
            </div>
        `;
    }}

    // Update bottom status
    const cid = clusterId !== undefined ? clusterId : currentClusterId;
    const statusEl = document.getElementById('graph-status');
    if (statusEl) {{
        statusEl.innerHTML = `Mode: Leiden C${{cid}} • Focused: <strong>${{geneId}}</strong> <span style="color:#38bdf8; margin-left:8px;">(Click again or double-click to view individual interactors →)</span>`;
    }}
}}

function unfocusClusterGene() {{
    clusterFocusedGene = null;
    if (cy) cy.elements().removeClass('focus highlighted dimmed');
    const banner = document.getElementById('cluster-gene-focus-banner');
    if (banner) banner.style.display = 'none';
    document.querySelectorAll('.partner').forEach(el => el.classList.remove('active-partner'));
    updateStatus();
}}

function handleClusterMemberClick(geneId) {{
    if (clusterFocusedGene === geneId) {{
        unfocusClusterGene();
        selectGene(geneId);
    }} else {{
        const node = cy.$id(geneId);
        if (node.length > 0) {{
            focusGeneInCluster(node, geneId, currentClusterId);
        }} else {{
            selectGene(geneId);
        }}
    }}
}}

function renderEgoNetwork(geneName) {{
    if (!cy || !geneName) return;
    unfocusClusterGene();
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
                jaccard: p.j,
                width: 1.2 + p.j * 4.5,
                opacity: Math.min(0.92, 0.30 + p.j * 0.9),
                color: getJaccardColor(p.j)
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
                        jaccard: pp.j,
                        width: 0.6 + pp.j * 1.5,
                        opacity: Math.min(0.18, 0.06 + pp.j * 0.20),
                        color: getJaccardColor(pp.j)
                    }}
                }});
            }}
        }}
    }});

    cy.batch(() => {{
        cy.elements().remove();
        cy.add(elements);
    }});
    runLayout(true);

    updateStatus();
}}


function runLayout(randomize) {{
    if (randomize === undefined) randomize = true;
    if (!cy || cy.nodes().length === 0) return;

    const layoutMode = document.getElementById('layout-select') ? document.getElementById('layout-select').value : 'cose';

    if (layoutMode === 'concentric') {{
        const concentricConfig = {{
            name: 'concentric',
            animate: true,
            animationDuration: 500,
            fit: randomize,
            padding: 60,
            startAngle: 3/2 * Math.PI,
            clockwise: true,
            equidistant: false,
            minNodeSpacing: 65,
            concentric: function(node) {{
                if (node.id() === selectedGene) return 10;
                return 1;
            }},
            levelWidth: function() {{ return 1; }}
        }};
        const layout = cy.layout(concentricConfig);
        if (randomize) {{
            layout.promiseOn('layoutstop').then(() => {{
                cy.animate({{ fit: {{ padding: 60 }} }}, {{ duration: 300 }});
            }});
        }}
        layout.run();
        return;
    }}

    // Force-directed (cose) - organic spread layout with label awareness
    const isEgo = Boolean(selectedGene && (currentMode === 'gene' || currentMode === 'network'));
    let intraEdges = null;
    let intraData = null;

    if (isEgo) {{
        // In ego networks, temporarily detach intra-partner edges during physics calculation
        // so the 180 secondary springs don't collapse nodes into an unreadable clump.
        intraEdges = cy.edges().filter(e => e.data('source') !== selectedGene && e.data('target') !== selectedGene);
        intraData = intraEdges.map(e => e.json());
        intraEdges.remove();
    }}

    const layoutConfig = {{
        name: 'cose',
        animate: false,
        randomize: randomize,
        nodeDimensionsIncludeLabels: true,
        componentSpacing: isEgo ? 130 : 100,
        nodeRepulsion: function(node) {{
            if (isEgo) return node.id() === selectedGene ? 20000000 : 8000000;
            return 2500000;
        }},
        nodeOverlap: 20,
        idealEdgeLength: function(edge) {{
            if (isEgo) {{
                const j = edge.data('jaccard') || 0.2;
                return Math.max(160, 310 - (j - 0.2) * 250);
            }}
            return 160;
        }},
        edgeElasticity: function(edge) {{ return isEgo ? 25 : 15; }},
        nestingFactor: 1.0,
        gravity: isEgo ? 0.04 : 0.06,
        numIter: 1400,
        initialTemp: 1000,
        coolingFactor: 0.98,
        minTemp: 1.0,
        fit: randomize,
        padding: 60
    }};

    const layout = cy.layout(layoutConfig);
    if (randomize) {{
        layout.promiseOn('layoutstop').then(() => {{
            cy.animate({{
                fit: {{ padding: 60 }}
            }}, {{ duration: 300 }});
        }});
    }}
    layout.promiseOn('layoutstop').then(() => {{
        if (isEgo && intraData && intraData.length > 0) {{
            cy.add(intraData);
            // Re-apply subtle styling to restored intra-partner edges
            cy.edges().forEach(e => {{
                if (e.data('source') !== selectedGene && e.data('target') !== selectedGene) {{
                    const j = e.data('jaccard') || 0.2;
                    e.style({{
                        'opacity': Math.min(0.18, 0.06 + j * 0.20),
                        'width': 0.6 + j * 1.5,
                        'line-color': getJaccardColor(j)
                    }});
                }}
            }});
        }}
        cy.animate({{ fit: {{ padding: 60 }} }}, {{ duration: 300 }});
    }});
    layout.run();
}}

function showSingleCluster(cid, highlightGene) {{
    if (!CLUSTERS[cid]) return;
    currentClusterId = cid;
    clusterFocusedGene = null;
    switchView('single_cluster');
    currentClusterId = cid;

    const members = CLUSTERS[cid];
    const color = getClusterColor(cid);
    const cname = CLUSTER_NAMES[cid] || `Cluster ${{cid}}`;

    const activeSet = getActiveGeneSet();
    const shown = members.filter(n => GM[n] !== undefined && (!activeSet || activeSet.has(n)));
    const shownSet = new Set(shown);

    document.getElementById('gene-info').innerHTML = `
        <a class="back-to-clusters" onclick="showAllClusters();">← All Leiden Clusters</a>
        <h2 style="color:${{color}};">C${{cid}}: ${{cname}}</h2>
        <div class="meta">${{shown.length}} genes • Jaccard &amp; Top-N filters inactive for clusters</div>
        <button class="btn btn-tree-add" style="margin-top:7px;" onclick="addClusterGenesToTree(${{cid}})" title="Add top members of this cluster to Species Tree">🌳 Add Cluster to Tree</button>
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

    // Add all intra-cluster edges (no Jaccard threshold — show full community structure)
    shown.forEach(name => {{
        const d = getGeneData(name);
        if (!d) return;
        for (const p of d.p) {{
            if (p.j > 0 && shownSet.has(p.n) && name < p.n) {{
                elements.push({{
                    group: 'edges',
                    data: {{
                        id: `${{name}}--${{p.n}}`,
                        source: name,
                        target: p.n,
                        jaccard: p.j,
                        width: 0.8 + p.j * 3.5,
                        opacity: Math.min(0.80, 0.20 + p.j * 0.8),
                        color: getJaccardColor(p.j)
                    }}
                }});
            }}
        }}
    }});

    cy.batch(() => {{
        cy.elements().remove();
        cy.add(elements);
    }});
    runLayout(true);

    // Populate sidebar with all cluster members
    let sideHtml = '';
    shown.forEach((m, i) => {{
        const isCil = ALL_CILIARY.has(m);
        sideHtml += `
        <div class="partner ${{m === highlightGene ? 'active-partner' : ''}}" data-gene="${{m}}" onclick="handleClusterMemberClick('${{m}}')">
            <span class="rank">#${{i + 1}}</span>
            <span class="pname in-graph">${{m}}</span>
            ${{isCil ? getCiliaBadgeHtml(m) : ''}}
            <span class="ploss">${{GM[m] || 0}}L</span>
            <button class="partner-tree-btn" onclick="event.stopPropagation(); addTreeGene('${{m}}');" title="Add ${{m}} to Species Tree">🌳</button>
            <a href="${{getUniProtUrl(m)}}" target="_blank" rel="noopener noreferrer" class="partner-ext-link" onclick="event.stopPropagation();">↗</a>
        </div>`;
    }});
    document.getElementById('partner-list').innerHTML = sideHtml;
    updateStatus();

    if (highlightGene) {{
        setTimeout(() => {{
            const targetNode = cy.$id(highlightGene);
            if (targetNode.length > 0) {{
                focusGeneInCluster(targetNode, highlightGene, cid);
            }}
        }}, 650);
    }}
}}

function addClusterGenesToTree(cid) {{
    if (!CLUSTERS[cid]) return;
    const members = CLUSTERS[cid].filter(n => GM[n] !== undefined);
    const topGenes = members.sort((a, b) => (GM[b] || 0) - (GM[a] || 0)).slice(0, 5);
    let addedCount = 0;
    topGenes.forEach(g => {{
        if (!TREE_SELECTED_GENES.includes(g)) {{
            TREE_SELECTED_GENES.push(g);
            addedCount++;
        }}
    }});
    renderTreeChips();
    if (TREE_LAYOUT) renderCircularTree();
    showTreeToast(`🌳 Added ${{addedCount}} genes from C${{cid}} to Species Tree (${{TREE_SELECTED_GENES.length}} on tree)`);
}}

function showAllClusters() {{
    currentClusterId = null;
    unfocusClusterGene();
    currentMode = 'all_clusters';
    selectedGene = null;
    document.getElementById('search').value = '';

    const btnNet = document.getElementById('btn-view-cy');
    const btnClusters = document.getElementById('btn-all-clusters-head');
    const btnTree = document.getElementById('btn-view-tree');
    const cyEl = document.getElementById('cy');
    const treeEl = document.getElementById('tree-panel');

    updateControlsForMode(true);

    if (btnNet) btnNet.classList.remove('active');
    if (btnClusters) btnClusters.classList.add('active');
    if (btnTree) btnTree.classList.remove('active');

    if (treeEl) treeEl.style.display = 'none';
    if (cyEl) cyEl.style.display = 'block';
    if (cy) cy.resize();

    const activeSet = getActiveGeneSet();
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

        const sorted = members.filter(n => GM[n] !== undefined && (!activeSet || activeSet.has(n))).map(n => ({{ n, l: GM[n] }})).sort((a, b) => b.l - a.l);
        const vis = sorted.slice(0, MAX_NODES);
        const nVis = vis.length;
        const radius = Math.max(45, Math.min(150, Math.sqrt(Math.max(1, nVis)) * 28));
        const visSet = new Set(vis.map(m => m.n));

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

        // Add intra-cluster edges to make community structure visible
        vis.forEach(m => {{
            const d = getGeneData(m.n);
            if (!d) return;
            for (const p of d.p) {{
                if (p.j > 0 && visSet.has(p.n) && m.n < p.n) {{
                    elements.push({{
                        group: 'edges',
                        data: {{
                            id: `${{m.n}}--${{p.n}}`,
                            source: m.n,
                            target: p.n,
                            jaccard: p.j,
                            width: 0.6 + p.j * 2.5,
                            opacity: Math.min(0.75, 0.18 + p.j * 0.7),
                            color: getJaccardColor(p.j)
                        }}
                    }});
                }}
            }}
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
    if (currentMode === 'tree') {{
        const geneCount = TREE_SELECTED_GENES.length;
        status.innerHTML = `Mode: Species Tree • 196 species • ${{geneCount}} gene${{geneCount === 1 ? '' : 's'}} on tree (${{TREE_TAX_LEVEL.toUpperCase()}} taxonomy)`;
    }} else if (currentMode === 'all_clusters') {{
        status.textContent = 'Mode: Leiden Modules • 80 clusters • ' + (cy ? cy.nodes().length : 0) + ' nodes';
    }} else if (currentMode === 'single_cluster') {{
        status.textContent = `Mode: Leiden Cluster C${{currentClusterId}} • ${{cy ? cy.nodes().length : 0}} nodes • ${{cy ? cy.edges().length : 0}} edges (click gene to focus, 2nd click opens interactors)`;
    }} else if (cy && selectedGene) {{
        const cid = GENE_CL[selectedGene];
        const cname = cid !== undefined ? (CLUSTER_NAMES[cid] || `Cluster ${{cid}}`) : '';
        status.innerHTML = `Mode: Network View • Focus: <strong>${{selectedGene}}</strong> (Cluster C${{cid}}: ${{cname}}) • ${{cy.nodes().length}} nodes • ${{cy.edges().length}} edges`;
    }} else if (cy) {{
        status.textContent = `Mode: Network View • ${{cy.nodes().length}} nodes • ${{cy.edges().length}} edges`;
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
document.getElementById('layout-select').addEventListener('change', function() {{
    runLayout(true);
}});

// ---- Circular Species Tree Visualizer (TCS Cladogram) ----
let TREE_LAYOUT = null;
let TREE_PRESENCE_BUFFER = null;
let TREE_GENE_IDX = {{}};
let TREE_SELECTED_GENES = [];
let TREE_TAX_LEVEL = 'kingdom';
let TREE_BRANCH_MODE = 'genes';
let TREE_BRANCH_LEN_MODE = 'cladogram'; // default: ignore branch lengths (like in iTOL)
const TREE_PALETTE = ['#d62728', '#1f77b4', '#2ca02c', '#d95f02', '#9467bd', '#17becf', '#e377c2', '#8c564b', '#bcbd22', '#17becf'];

let treeZoomScale = 1.0;
let treePanX = 0;
let treePanY = 0;
let isTreePanning = false;
let treePanStartX = 0;
let treePanStartY = 0;

function openTreeView() {{
    switchView('tree');
}}

function closeTreeView() {{
    switchView('network');
}}

function setTreeTaxLevel(lvl) {{
    TREE_TAX_LEVEL = lvl;
    const sel = document.getElementById('tree-tax-select');
    if (sel && sel.value !== lvl) sel.value = lvl;
    renderCircularTree();
}}

function setTreeBranchMode(mode) {{
    TREE_BRANCH_MODE = mode;
    renderCircularTree();
}}

function setTreeBranchLength(mode) {{
    TREE_BRANCH_LEN_MODE = mode;
    renderCircularTree();
}}

async function initTreeView() {{
    if (!TREE_LAYOUT) {{
        document.getElementById('tree-loading').style.display = 'flex';
        try {{
            const [layoutResp, presenceResp] = await Promise.all([
                fetch('tree_layout.json'),
                fetch('tree_presence.bin')
            ]);
            if (!layoutResp.ok || !presenceResp.ok) throw new Error('Failed to load tree assets');
            TREE_LAYOUT = await layoutResp.json();
            const pBuf = await presenceResp.arrayBuffer();
            // Header: "DLTP", uint32 n_genes, uint16 n_leaves (10 bytes)
            TREE_PRESENCE_BUFFER = new Uint8Array(pBuf, 10);
            TREE_LAYOUT.gene_names.forEach((name, i) => {{ TREE_GENE_IDX[name] = i; }});
            
            setupTreePanZoom();
            setupTreeSearch();
        }} catch (err) {{
            console.error('Error loading tree:', err);
            document.getElementById('tree-loading').innerHTML = '<span style="color:#f87171;">Error loading tree data: ' + err.message + '</span>';
            return;
        }}
    }}
    document.getElementById('tree-loading').style.display = 'none';
    renderTreeChips();
    renderCircularTree();
}}

function getTreeGenePresence(geneName) {{
    const nLeaves = TREE_LAYOUT.n_leaves;
    if (TREE_GENE_IDX[geneName] === undefined || !TREE_PRESENCE_BUFFER) {{
        return new Uint8Array(nLeaves);
    }}
    const offset = TREE_GENE_IDX[geneName] * 25;
    const res = new Uint8Array(nLeaves);
    for (let i = 0; i < nLeaves; i++) {{
        const byteI = offset + Math.floor(i / 8);
        const bitI = i % 8;
        if (TREE_PRESENCE_BUFFER[byteI] & (1 << bitI)) {{
            res[i] = 1;
        }}
    }}
    return res;
}}

function renderTreeChips() {{
    const container = document.getElementById('tree-chips-container');
    if (!container) return;
    if (TREE_SELECTED_GENES.length === 0) {{
        container.innerHTML = '<span style="font-size:11px; color:#64748b; font-style:italic;">No genes selected (add from sidebar or search above)</span>';
        return;
    }}
    const isDark = getEffectiveTreeTheme() === 'dark';
    const chipBg = isDark ? '#161d31' : '#f1f5f9';
    const textCol = isDark ? '#f8fafc' : '#0f172a';
    const closeCol = isDark ? '#94a3b8' : '#64748b';
    container.innerHTML = TREE_SELECTED_GENES.map((g, idx) => {{
        const col = TREE_PALETTE[idx % TREE_PALETTE.length];
        return `
            <div class="tree-gene-chip" style="display:inline-flex; align-items:center; gap:5px; background:${{chipBg}}; border:1px solid ${{col}}; border-radius:14px; padding:2px 8px; font-size:11px; color:${{textCol}};">
                <span style="width:8px; height:8px; border-radius:50%; background:${{col}};"></span>
                <strong>${{g}}</strong>
                <span style="cursor:pointer; color:${{closeCol}}; margin-left:3px; font-weight:700;" onclick="removeTreeGene('${{g}}')">&times;</span>
            </div>
        `;
    }}).join('');
}}

function addTreeGene(geneName) {{
    const clean = geneName.trim().toUpperCase();
    if (!clean) return;
    if (TREE_SELECTED_GENES.includes(clean)) {{
        showTreeToast(`${{clean}} is already on the tree`);
        return;
    }}
    if (TREE_LAYOUT && TREE_GENE_IDX[clean] === undefined) {{
        showTreeToast(`Gene "${{clean}}" not found in tree dataset`);
        return;
    }}
    TREE_SELECTED_GENES.push(clean);
    renderTreeChips();
    if (TREE_LAYOUT) renderCircularTree();
    showTreeToast(`🌳 Added ${{clean}} to Species Tree (${{TREE_SELECTED_GENES.length}} on tree)`);
}}

function removeTreeGene(geneName) {{
    TREE_SELECTED_GENES = TREE_SELECTED_GENES.filter(g => g !== geneName);
    renderTreeChips();
    if (TREE_LAYOUT) renderCircularTree();
    showTreeToast(`Removed ${{geneName}} from tree`);
}}

function resetTreeGenes(genes) {{
    TREE_SELECTED_GENES = [...(genes || [])];
    renderTreeChips();
    if (TREE_LAYOUT) renderCircularTree();
    if (TREE_SELECTED_GENES.length === 0) {{
        showTreeToast('Cleared all genes on tree');
    }}
}}

function loadCurrentNetworkGenesInTree() {{
    if (selectedGene && (TREE_GENE_IDX[selectedGene] !== undefined || !TREE_LAYOUT)) {{
        const list = [selectedGene];
        const gData = getGeneData(selectedGene);
        if (gData && gData.p) {{
            for (const partner of gData.p) {{
                if (!list.includes(partner.n)) {{
                    list.push(partner.n);
                    if (list.length >= 3) break;
                }}
            }}
        }}
        resetTreeGenes(list);
        showTreeToast(`Loaded ${{list.join(', ')}} onto tree`);
    }} else {{
        showTreeToast('Select a gene in the network first to load it and its partners');
    }}
}}

function blendTreeColors(presentIndices, isDark) {{
    if (!presentIndices || presentIndices.length === 0) {{
        return isDark ? '#334155' : '#cbd5e1'; // Dark slate in dark mode, light grey in light mode
    }}
    if (presentIndices.length === 1) return TREE_PALETTE[presentIndices[0] % TREE_PALETTE.length];
    if (presentIndices.length === TREE_SELECTED_GENES.length && TREE_SELECTED_GENES.length > 1) {{
        return isDark ? '#ffffff' : '#111827'; // Crisp white in dark mode, dark navy in light mode
    }}

    if (presentIndices.length === 2 && TREE_SELECTED_GENES.length >= 2) {{
        const i1 = presentIndices[0], i2 = presentIndices[1];
        if ((i1 === 0 && i2 === 1) || (i1 === 1 && i2 === 0)) return '#a855f7';
        if ((i1 === 0 && i2 === 2) || (i1 === 2 && i2 === 0)) return '#f97316';
        if ((i1 === 1 && i2 === 2) || (i1 === 2 && i2 === 1)) return '#06b6d4';
    }}

    // Subtractive / additive blend
    let r = 0, g = 0, b = 0;
    presentIndices.forEach(idx => {{
        const hex = TREE_PALETTE[idx % TREE_PALETTE.length];
        const cR = parseInt(hex.slice(1, 3), 16);
        const cG = parseInt(hex.slice(3, 5), 16);
        const cB = parseInt(hex.slice(5, 7), 16);
        r += cR; g += cG; b += cB;
    }});
    const factor = isDark ? 1.05 : 0.85;
    r = Math.min(255, Math.floor((r / presentIndices.length) * factor));
    g = Math.min(255, Math.floor((g / presentIndices.length) * factor));
    b = Math.min(255, Math.floor((b / presentIndices.length) * factor));
    return `rgb(${{r}},${{g}},${{b}})`;
}}

function renderCircularTree() {{
    if (!TREE_LAYOUT) return;
    const svg = document.getElementById('tree-svg');
    if (!svg) return;

    const isDark = getEffectiveTreeTheme() === 'dark';

    const SIZE = 1000;
    const CX = SIZE / 2;
    const CY = SIZE / 2;
    const SPAN = 360.0 - TREE_LAYOUT.gap_deg;
    const nLeaves = TREE_LAYOUT.n_leaves;

    const presenceVectors = TREE_SELECTED_GENES.map(g => getTreeGenePresence(g));
    const cladeBlocks = (TREE_LAYOUT.clade_levels && TREE_LAYOUT.clade_levels[TREE_TAX_LEVEL]) || TREE_LAYOUT.clade_blocks;

    // Calculate node colors
    const nodeColors = {{}};
    const idToNode = {{}};
    TREE_LAYOUT.nodes.forEach(n => {{ idToNode[n.id] = n; }});

    const defaultNodeCol = isDark ? '#475569' : '#94a3b8';

    if (TREE_BRANCH_MODE === 'taxonomy' || TREE_SELECTED_GENES.length === 0) {{
        // Color branches by taxonomic clade of descendant leaves
        TREE_LAYOUT.nodes.forEach(node => {{
            if (node.leaf) {{
                const lInfo = TREE_LAYOUT.leaf_taxonomies ? TREE_LAYOUT.leaf_taxonomies[node.leaf_idx] : null;
                nodeColors[node.id] = lInfo ? (lInfo[TREE_TAX_LEVEL + '_color'] || defaultNodeCol) : defaultNodeCol;
            }} else {{
                const counts = {{}};
                const colors = {{}};
                node.leaves.forEach(lIdx => {{
                    const lInfo = TREE_LAYOUT.leaf_taxonomies ? TREE_LAYOUT.leaf_taxonomies[lIdx] : null;
                    if (!lInfo) return;
                    const c = lInfo[TREE_TAX_LEVEL];
                    counts[c] = (counts[c] || 0) + 1;
                    colors[c] = lInfo[TREE_TAX_LEVEL + '_color'] || defaultNodeCol;
                }});
                let maxCount = 0;
                let domClade = null;
                for (const [c, cnt] of Object.entries(counts)) {{
                    if (cnt > maxCount) {{
                        maxCount = cnt;
                        domClade = c;
                    }}
                }}
                if (domClade && (maxCount / node.leaves.length >= 0.70)) {{
                    nodeColors[node.id] = colors[domClade];
                }} else {{
                    nodeColors[node.id] = defaultNodeCol;
                }}
            }}
        }});
    }} else {{
        // Color branches by gene presence combinations (Dollo parsimony)
        TREE_LAYOUT.nodes.forEach(node => {{
            const presentIndices = [];
            for (let gI = 0; gI < presenceVectors.length; gI++) {{
                const pVec = presenceVectors[gI];
                if (node.leaves.some(lIdx => pVec[lIdx] === 1)) {{
                    presentIndices.push(gI);
                }}
            }}
            nodeColors[node.id] = blendTreeColors(presentIndices, isDark);
        }});
    }}

    const svgParts = [];
    const taxLabel = TREE_TAX_LEVEL === 'tcs' ? 'TCS Clades' : (TREE_TAX_LEVEL === 'detailed' ? 'Phylum' : TREE_TAX_LEVEL.charAt(0).toUpperCase() + TREE_TAX_LEVEL.slice(1));
    const titleText = TREE_SELECTED_GENES.length > 0
        ? (TREE_BRANCH_MODE === 'taxonomy'
            ? `${{TREE_SELECTED_GENES.join(', ')}} presence across eukaryotes (${{taxLabel}} taxonomy)`
            : `${{TREE_SELECTED_GENES.join(', ')}} presence across eukaryotes`)
        : `Eukaryotic Species Tree (196 species • ${{taxLabel}} taxonomy)`;
    const titleColor = isDark ? '#f8fafc' : '#111827';
    svgParts.push(`<text x="500" y="32" text-anchor="middle" font-size="16" font-weight="700" fill="${{titleColor}}">${{titleText}}</text>`);

    // Dimensions
    const R_ROOT = TREE_LAYOUT.r_root;
    const R_TREE = TREE_LAYOUT.r_tree;
    const R_TRACKS_START = R_TREE + 12;
    const TRACK_WIDTH = 10;
    const R_TRACKS_END = R_TRACKS_START + TREE_SELECTED_GENES.length * (TRACK_WIDTH + 2);
    const R_CLADE_ARC = R_TRACKS_END + 14;

    function pt(r, angDeg) {{
        const rad = (angDeg * Math.PI) / 180.0;
        return [CX + r * Math.cos(rad), CY + r * Math.sin(rad)];
    }}

    // 1. Clade Background Wedges
    const wedgeOpacity = isDark ? '0.14' : '0.08';
    const stepA = (TREE_LAYOUT.gap_deg && TREE_LAYOUT.gap_deg > 0) ? (SPAN / (nLeaves - 1)) : (SPAN / nLeaves);
    const dA = stepA * 0.48;

    cladeBlocks.forEach(block => {{
        const sIdx = block.start_idx;
        const eIdx = block.end_idx;
        const col = block.color || (isDark ? '#475569' : '#94a3b8');
        const a1 = TREE_LAYOUT.leaf_angles[sIdx] + dA;
        const a2 = TREE_LAYOUT.leaf_angles[eIdx] - dA;

        const rad1 = (a1 * Math.PI) / 180.0;
        const rad2 = (a2 * Math.PI) / 180.0;
        const rMax = R_CLADE_ARC + 32;

        const x1 = (CX + R_ROOT * Math.cos(rad1)).toFixed(2);
        const y1 = (CY + R_ROOT * Math.sin(rad1)).toFixed(2);
        const x2 = (CX + rMax * Math.cos(rad1)).toFixed(2);
        const y2 = (CY + rMax * Math.sin(rad1)).toFixed(2);
        const x3 = (CX + rMax * Math.cos(rad2)).toFixed(2);
        const y3 = (CY + rMax * Math.sin(rad2)).toFixed(2);
        const x4 = (CX + R_ROOT * Math.cos(rad2)).toFixed(2);
        const y4 = (CY + R_ROOT * Math.sin(rad2)).toFixed(2);

        svgParts.push(`<path d="M ${{x1}} ${{y1}} L ${{x2}} ${{y2}} A ${{rMax}} ${{rMax}} 0 0 0 ${{x3}} ${{y3}} L ${{x4}} ${{y4}} Z" fill="${{col}}" fill-opacity="${{wedgeOpacity}}" stroke="none" />`);
    }});

    // 2. Tree Branches
    const branchWidth = isDark ? '1.8' : '1.7';
    TREE_LAYOUT.nodes.forEach(node => {{
        if (node.p === null || node.p === undefined) return;
        const parent = idToNode[node.p];
        const rP = (TREE_BRANCH_LEN_MODE === 'phylogram' && parent.r_phylo !== undefined) ? parent.r_phylo : (parent.r_clad !== undefined ? parent.r_clad : parent.r);
        const aP = parent.a;
        const rC = (TREE_BRANCH_LEN_MODE === 'phylogram' && node.r_phylo !== undefined) ? node.r_phylo : (node.r_clad !== undefined ? node.r_clad : node.r);
        const aC = node.a;
        const col = nodeColors[node.id];

        const [xP, yP] = pt(rP, aP);
        const [xArc, yArc] = pt(rP, aC);
        const [xC, yC] = pt(rC, aC);

        const sweep = aC < aP ? 0 : 1;
        const pathD = `M ${{xP.toFixed(2)}} ${{yP.toFixed(2)}} A ${{rP.toFixed(2)}} ${{rP.toFixed(2)}} 0 0 ${{sweep}} ${{xArc.toFixed(2)}} ${{yArc.toFixed(2)}} L ${{xC.toFixed(2)}} ${{yC.toFixed(2)}}`;
        svgParts.push(`<path d="${{pathD}}" fill="none" stroke="${{col}}" stroke-width="${{branchWidth}}" stroke-linecap="round" />`);
    }});

    // 3. Concentric Tracks
    const absentFill = isDark ? '#151d30' : '#f8fafc';
    const absentStroke = isDark ? '#263350' : '#e2e8f0';

    TREE_SELECTED_GENES.forEach((gName, gI) => {{
        const rInner = R_TRACKS_START + gI * (TRACK_WIDTH + 2);
        const rOuter = rInner + TRACK_WIDTH;
        const col = TREE_PALETTE[gI % TREE_PALETTE.length];
        const pVec = presenceVectors[gI];

        TREE_LAYOUT.leaf_species.forEach((sp, leafI) => {{
            const isPres = pVec[leafI] === 1;
            const ang = TREE_LAYOUT.leaf_angles[leafI];
            const a1 = ang + dA;
            const a2 = ang - dA;
            const rad1 = (a1 * Math.PI) / 180.0;
            const rad2 = (a2 * Math.PI) / 180.0;

            const fillCol = isPres ? col : absentFill;
            const strokeCol = isPres ? col : absentStroke;

            const x1 = (CX + rInner * Math.cos(rad1)).toFixed(2);
            const y1 = (CY + rInner * Math.sin(rad1)).toFixed(2);
            const x2 = (CX + rOuter * Math.cos(rad1)).toFixed(2);
            const y2 = (CY + rOuter * Math.sin(rad1)).toFixed(2);
            const x3 = (CX + rOuter * Math.cos(rad2)).toFixed(2);
            const y3 = (CY + rOuter * Math.sin(rad2)).toFixed(2);
            const x4 = (CX + rInner * Math.cos(rad2)).toFixed(2);
            const y4 = (CY + rInner * Math.sin(rad2)).toFixed(2);

            const pathD = `M ${{x1}} ${{y1}} L ${{x2}} ${{y2}} A ${{rOuter}} ${{rOuter}} 0 0 0 ${{x3}} ${{y3}} L ${{x4}} ${{y4}} A ${{rInner}} ${{rInner}} 0 0 1 ${{x1}} ${{y1}} Z`;
            svgParts.push(`<path class="tree-tile" data-sp="${{sp}}" data-gene="${{gName}}" data-pres="${{isPres ? 1 : 0}}" d="${{pathD}}" fill="${{fillCol}}" stroke="${{strokeCol}}" stroke-width="0.35" style="cursor:pointer;" />`);
        }});
    }});

    // 4. Clade Outer Arcs and Labels
    cladeBlocks.forEach(block => {{
        const sIdx = block.start_idx;
        const eIdx = block.end_idx;
        const cname = block.clade;
        const col = block.color || (isDark ? '#94a3b8' : '#64748b');

        const a1 = TREE_LAYOUT.leaf_angles[sIdx] + dA;
        const a2 = TREE_LAYOUT.leaf_angles[eIdx] - dA;
        const rad1 = (a1 * Math.PI) / 180.0;
        const rad2 = (a2 * Math.PI) / 180.0;

        const xS = (CX + R_CLADE_ARC * Math.cos(rad1)).toFixed(2);
        const yS = (CY + R_CLADE_ARC * Math.sin(rad1)).toFixed(2);
        const xE = (CX + R_CLADE_ARC * Math.cos(rad2)).toFixed(2);
        const yE = (CY + R_CLADE_ARC * Math.sin(rad2)).toFixed(2);

        svgParts.push(`<path d="M ${{xS}} ${{yS}} A ${{R_CLADE_ARC}} ${{R_CLADE_ARC}} 0 0 0 ${{xE}} ${{yE}}" fill="none" stroke="${{col}}" stroke-width="5.5" stroke-linecap="round" />`);

        // Skip label for tiny single-species sectors to avoid crowded/isolated labels
        const blockSize = (eIdx - sIdx) + 1;
        if (blockSize < 2) return;

        const midAng = (a1 + a2) / 2.0;
        const rLabel = R_CLADE_ARC + 14;
        const [lx, ly] = pt(rLabel, midAng);

        const normRot = ((midAng % 360) + 360) % 360;
        let rot = normRot;
        let anchor = 'start';
        if (normRot > 90 && normRot < 270) {{
            rot = normRot + 180;
            anchor = 'end';
        }}

        const fontSize = cladeBlocks.length > 40 ? 7.5 : 8.5;
        svgParts.push(`<text x="${{lx.toFixed(2)}}" y="${{ly.toFixed(2)}}" transform="rotate(${{rot.toFixed(1)}}, ${{lx.toFixed(2)}}, ${{ly.toFixed(2)}})" font-size="${{fontSize}}" font-weight="700" fill="${{col}}" text-anchor="${{anchor}}" alignment-baseline="middle">${{cname}}</text>`);
    }});

    // 5. Dynamic Legend in Top-Left
    const legTitleCol = isDark ? '#f8fafc' : '#111827';
    const legSubCol = isDark ? '#94a3b8' : '#6b7280';
    const legTextCol = isDark ? '#cbd5e1' : '#374151';

    svgParts.push('<g transform="translate(35, 45)">');
    if (TREE_BRANCH_MODE === 'taxonomy' || TREE_SELECTED_GENES.length === 0) {{
        svgParts.push(`<text x="0" y="0" font-size="11" font-weight="700" fill="${{legTitleCol}}">Taxonomy: ${{taxLabel}}</text>`);
        svgParts.push(`<text x="0" y="14" font-size="9" fill="${{legSubCol}}">(clade color coding)</text>`);

        const seenClades = new Set();
        let lI = 0;
        cladeBlocks.forEach(block => {{
            if (seenClades.has(block.clade) || lI >= 12) return;
            seenClades.add(block.clade);
            const y = 30 + lI * 15;
            svgParts.push(`<rect x="0" y="${{y}}" width="18" height="8" rx="2" fill="${{block.color}}" />`);
            svgParts.push(`<text x="24" y="${{y + 7}}" font-size="9" fill="${{legTextCol}}">${{block.clade}}</text>`);
            lI++;
        }});
        if (seenClades.size > 12) {{
            const y = 30 + lI * 15;
            svgParts.push(`<text x="0" y="${{y + 7}}" font-size="8.5" fill="#94a3b8">+ ${{seenClades.size - 12}} more...</text>`);
        }}
    }} else {{
        svgParts.push(`<text x="0" y="0" font-size="11" font-weight="700" fill="${{legTitleCol}}">Branch Color: Gene Presence</text>`);
        svgParts.push(`<text x="0" y="14" font-size="9" fill="${{legSubCol}}">(presence combinations)</text>`);

        const noneCol = isDark ? '#334155' : '#cbd5e1';
        const allCol = isDark ? '#ffffff' : '#111827';

        const uniqueCols = new Map();
        uniqueCols.set(noneCol, 'None present');
        if (TREE_SELECTED_GENES.length === 1) {{
            uniqueCols.set(TREE_PALETTE[0], TREE_SELECTED_GENES[0]);
        }} else if (TREE_SELECTED_GENES.length === 2) {{
            uniqueCols.set(TREE_PALETTE[0], TREE_SELECTED_GENES[0]);
            uniqueCols.set(TREE_PALETTE[1], TREE_SELECTED_GENES[1]);
            uniqueCols.set(allCol, `${{TREE_SELECTED_GENES[0]}} + ${{TREE_SELECTED_GENES[1]}}`);
        }} else if (TREE_SELECTED_GENES.length === 3) {{
            uniqueCols.set(TREE_PALETTE[0], TREE_SELECTED_GENES[0]);
            uniqueCols.set(TREE_PALETTE[1], TREE_SELECTED_GENES[1]);
            uniqueCols.set(TREE_PALETTE[2], TREE_SELECTED_GENES[2]);
            uniqueCols.set('#a855f7', `${{TREE_SELECTED_GENES[0]}} + ${{TREE_SELECTED_GENES[1]}}`);
            uniqueCols.set('#f97316', `${{TREE_SELECTED_GENES[0]}} + ${{TREE_SELECTED_GENES[2]}}`);
            uniqueCols.set('#06b6d4', `${{TREE_SELECTED_GENES[1]}} + ${{TREE_SELECTED_GENES[2]}}`);
            uniqueCols.set(allCol, 'All 3 genes');
        }} else {{
            TREE_SELECTED_GENES.forEach((g, idx) => {{
                uniqueCols.set(TREE_PALETTE[idx % TREE_PALETTE.length], g);
            }});
            if (TREE_SELECTED_GENES.length > 1) {{
                uniqueCols.set(allCol, `All ${{TREE_SELECTED_GENES.length}} genes`);
            }}
        }}

        let lI = 0;
        uniqueCols.forEach((label, col) => {{
            const y = 30 + lI * 16;
            svgParts.push(`<rect x="0" y="${{y}}" width="18" height="8" rx="2" fill="${{col}}" />`);
            svgParts.push(`<text x="24" y="${{y + 7}}" font-size="9.5" fill="${{legTextCol}}">${{label}}</text>`);
            lI++;
        }});
    }}
    svgParts.push('</g>');

    svg.innerHTML = svgParts.join('');
    setupTreeTooltips();
}}

function setupTreePanZoom() {{
    const viewport = document.getElementById('tree-viewport');
    const wrapper = document.getElementById('tree-canvas-wrapper');
    if (!viewport || !wrapper) return;

    viewport.addEventListener('wheel', (e) => {{
        e.preventDefault();
        const factor = e.deltaY < 0 ? 1.15 : 0.85;
        zoomTree(factor);
    }}, {{ passive: false }});

    viewport.addEventListener('mousedown', (e) => {{
        if (e.target.closest('button') || e.target.closest('select') || e.target.closest('input')) return;
        isTreePanning = true;
        treePanStartX = e.clientX - treePanX;
        treePanStartY = e.clientY - treePanY;
        viewport.style.cursor = 'grabbing';
    }});

    window.addEventListener('mousemove', (e) => {{
        if (!isTreePanning) return;
        treePanX = e.clientX - treePanStartX;
        treePanY = e.clientY - treePanStartY;
        applyTreeTransform();
    }});

    window.addEventListener('mouseup', () => {{
        isTreePanning = false;
        if (viewport) viewport.style.cursor = 'grab';
    }});
}}

function zoomTree(factor) {{
    treeZoomScale = Math.max(0.5, Math.min(6.0, treeZoomScale * factor));
    applyTreeTransform();
}}

function resetTreeZoom() {{
    treeZoomScale = 1.0;
    treePanX = 0;
    treePanY = 0;
    applyTreeTransform();
}}

function applyTreeTransform() {{
    const wrapper = document.getElementById('tree-canvas-wrapper');
    if (wrapper) {{
        wrapper.style.transform = `translate(${{treePanX}}px, ${{treePanY}}px) scale(${{treeZoomScale}})`;
    }}
}}

function setupTreeSearch() {{
    const input = document.getElementById('tree-gene-input');
    const sugBox = document.getElementById('tree-gene-suggestions');
    if (!input || !sugBox) return;

    input.addEventListener('input', function() {{
        const q = this.value.trim().toUpperCase();
        if (!q || !TREE_LAYOUT) {{ sugBox.style.display = 'none'; return; }}
        const matches = TREE_LAYOUT.gene_names.filter(g => g.toUpperCase().includes(q)).slice(0, 15);
        if (matches.length === 0) {{ sugBox.style.display = 'none'; return; }}
        sugBox.innerHTML = matches.map(m => `
            <div style="padding:6px 10px; cursor:pointer; font-size:12px; border-bottom:1px solid #1e293b; color:#cbd5e1;" onmouseover="this.style.background='#1e293b'" onmouseout="this.style.background='transparent'" onclick="addTreeGene('${{m}}'); document.getElementById('tree-gene-suggestions').style.display='none'; document.getElementById('tree-gene-input').value='';">
                ${{m}}
            </div>
        `).join('');
        sugBox.style.display = 'block';
    }});

    input.addEventListener('keydown', function(e) {{
        if (e.key === 'Enter') {{
            const q = this.value.trim().toUpperCase();
            if (q && TREE_GENE_IDX[q] !== undefined) {{
                addTreeGene(q);
                sugBox.style.display = 'none';
                this.value = '';
            }}
        }}
    }});

    document.addEventListener('click', (e) => {{
        if (!e.target.closest('#tree-search-wrap') && !e.target.closest('#tree-gene-input')) {{
            sugBox.style.display = 'none';
        }}
    }});
}}

function setupTreeTooltips() {{
    const tooltip = document.getElementById('tree-tooltip');
    const tiles = document.querySelectorAll('.tree-tile');
    const isDark = getEffectiveTreeTheme() === 'dark';
    tiles.forEach(tile => {{
        tile.addEventListener('mouseenter', (e) => {{
            const sp = tile.getAttribute('data-sp');
            const leafIdx = TREE_LAYOUT.leaf_species.indexOf(sp);
            const taxInfo = (TREE_LAYOUT.leaf_taxonomies && TREE_LAYOUT.leaf_taxonomies[leafIdx]) || {{}};
            const curVal = taxInfo[TREE_TAX_LEVEL] || taxInfo.clade || '';

            const titleCol = isDark ? '#f8fafc' : '#0f172a';
            const subCol = isDark ? '#94a3b8' : '#64748b';
            const taxCol = isDark ? '#38bdf8' : '#0284c7';
            const borderCol = isDark ? '#334155' : '#e2e8f0';
            const presCol = isDark ? '#4ade80' : '#16a34a';
            const absCol = isDark ? '#94a3b8' : '#64748b';

            let html = `<strong style="color:${{titleCol}}; font-size:12px;">${{taxInfo.sci_name || sp.replace(/_/g, ' ')}}</strong><br>`;
            const taxName = TREE_TAX_LEVEL === 'tcs' ? 'TCS CLADE' : (TREE_TAX_LEVEL === 'detailed' ? 'PHYLUM' : TREE_TAX_LEVEL.toUpperCase());
            if (curVal) {{
                html += `<span style="color:${{taxCol}}; font-size:10.5px;">${{taxName}}: <strong>${{curVal}}</strong></span><br>`;
            }}
            if (taxInfo.supergroup && taxInfo.phylum) {{
                html += `<span style="color:${{subCol}}; font-size:10px;">${{taxInfo.supergroup}} &rarr; ${{taxInfo.kingdom}} &rarr; ${{taxInfo.phylum}}</span><br>`;
            }}
            html += `<div style="margin-top:6px; border-top:1px solid ${{borderCol}}; padding-top:4px;">`;
            TREE_SELECTED_GENES.forEach((gName, gI) => {{
                const isP = getTreeGenePresence(gName)[leafIdx] === 1;
                const col = TREE_PALETTE[gI % TREE_PALETTE.length];
                html += `
                    <div style="display:flex; justify-content:space-between; gap:12px; font-size:10.5px;">
                        <span style="color:${{col}}; font-weight:600;">${{gName}}:</span>
                        <span style="color:${{isP ? presCol : absCol}}; font-weight:600;">${{isP ? '● Present (1)' : '○ Absent (0)'}}</span>
                    </div>
                `;
            }});
            html += `</div>`;
            tooltip.innerHTML = html;
            tooltip.style.display = 'block';
        }});
        tile.addEventListener('mousemove', (e) => {{
            tooltip.style.left = (e.clientX + 14) + 'px';
            tooltip.style.top = (e.clientY + 14) + 'px';
        }});
        tile.addEventListener('mouseleave', () => {{
            tooltip.style.display = 'none';
        }});
    }});
}}

function exportTreeSvg() {{
    const svg = document.getElementById('tree-svg');
    if (!svg) return;
    const blob = new Blob([svg.outerHTML], {{ type: 'image/svg+xml;charset=utf-8' }});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `species_tree_${{TREE_SELECTED_GENES.join('_')}}.svg`;
    a.click();
    URL.revokeObjectURL(url);
}}



document.getElementById('toggle-labels').addEventListener('change', function() {{
    const show = this.checked;
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
    if ((currentMode === 'gene' || currentMode === 'network') && selectedGene) renderEgoNetwork(selectedGene);
}});

document.getElementById('topn').addEventListener('input', function() {{
    document.getElementById('topn-val').textContent = this.value;
    if (selectedGene) renderSidebar(selectedGene);
    if ((currentMode === 'gene' || currentMode === 'network') && selectedGene) renderEgoNetwork(selectedGene);
}});

document.getElementById('toggle-topn-max').addEventListener('change', function() {{
    document.getElementById('topn').disabled = this.checked;
    document.getElementById('topn-val').textContent = this.checked ? 'Max' : document.getElementById('topn').value;
    if (selectedGene) renderSidebar(selectedGene);
    if ((currentMode === 'gene' || currentMode === 'network') && selectedGene) renderEgoNetwork(selectedGene);
}});

document.getElementById('gene-filter').addEventListener('change', function() {{
    geneFilterMode = this.value;
    if (selectedGene) renderSidebar(selectedGene);
    if ((currentMode === 'gene' || currentMode === 'network') && selectedGene) renderEgoNetwork(selectedGene);
    if (currentMode === 'single_cluster') showSingleCluster(currentClusterId);
    if (currentMode === 'all_clusters') showAllClusters();
}});

// Init
window.addEventListener('DOMContentLoaded', () => {{
    applySiteTheme(SITE_THEME);
    initCy();
    loadPartnersGraph();
    showAllClusters();
}});

</script>
</body>
</html>
"""
    return html_code

if __name__ == "__main__":
    main()
