"""Circular EggNOG species tree coloured by taxonomic domain (superkingdom).

Reads tree_layout_eggnog.json (leaf angles, node radii, per-leaf taxonomy) and
draws the same radial layout as the explorer. Branches are coloured by domain
when the whole clade belongs to one domain, otherwise grey; the outer ring
marks each leaf's domain.

  python3 figures/eggnog_domains/make_domain_tree.py
"""
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
LAYOUT = ROOT / "tree_layout_eggnog.json"
OUT = Path(__file__).resolve().parent / "eggnog_domains_tree.png"

PALETTE = {"Bacteria": "#3b82f6", "Archaea": "#f59e0b", "Eukaryota": "#16a34a", "Viruses": "#a855f7",
           "Unclassified": "#9ca3af"}
GREY = "#6b7280"


def main():
    t = json.load(open(LAYOUT))
    leaf_dom = [tx.get("domain") or "Unclassified" for tx in t["leaf_taxonomies"]]
    nodes = t["nodes"]
    print("domains:", Counter(leaf_dom).most_common())

    # a node is single-domain if all its leaves [l0, l1] share one domain
    def clade_domain(n):
        seen = set(leaf_dom[n["l0"]:n["l1"] + 1])
        return seen.pop() if len(seen) == 1 else None

    fig = plt.figure(figsize=(12, 12), dpi=200, facecolor="white")
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96], aspect="equal")
    ax.set_xlim(-340, 340); ax.set_ylim(-340, 340); ax.axis("off")

    def xy(r, a):
        rad = math.radians(a)
        return r * math.cos(rad), r * math.sin(rad)

    for n in nodes:
        if n["p"] is None:
            continue
        p = nodes[n["p"]]
        col = PALETTE.get(clade_domain(n), GREY) if clade_domain(n) else GREY
        # radial branch from parent radius to this node
        x0, y0 = xy(p["r"], n["a"]); x1, y1 = xy(n["r"], n["a"])
        ax.plot([x0, x1], [y0, y1], color=col, lw=0.5)
    for n in nodes:
        if n["leaf"] or not n["c"]:
            continue
        kids = [nodes[c] for c in n["c"]]
        angs = [k["a"] for k in kids]
        # arc at this node's radius joining the children
        a0, a1 = min(angs), max(angs)
        th = [a0 + (a1 - a0) * i / 40 for i in range(41)]
        pts = [xy(n["r"], a) for a in th]
        dom = clade_domain(n)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=PALETTE.get(dom, GREY) if dom else GREY, lw=0.5)

    # outer ring: one thin bar per leaf, coloured by domain
    for n in nodes:
        if n["leaf"] and n["leaf_idx"] is not None:
            i = n["leaf_idx"]
            r0, r1 = 305, 322
            x0, y0 = xy(r0, n["a"]); x1, y1 = xy(r1, n["a"])
            ax.plot([x0, x1], [y0, y1], color=PALETTE.get(leaf_dom[i], GREY), lw=0.35)

    handles = [Patch(color=c, label=f"{d} ({sum(1 for x in leaf_dom if x == d)})")
               for d, c in PALETTE.items() if any(x == d for x in leaf_dom)]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=11, title="Domain (species)")
    fig.text(0.5, 0.975, "EggNOG species tree coloured by domain", ha="center", fontsize=16, weight="bold")
    fig.savefig(OUT, facecolor="white")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
