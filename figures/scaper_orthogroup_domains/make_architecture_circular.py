"""Circular TCS tree (SCAPER orthogroup species) with each species' SCAPER
protein drawn to scale as a radial bar: grey = protein length, coloured
segments = domains at their residue positions. Taxa are labelled and marked
with their supergroup colour.

  python3 figures/scaper_orthogroup_domains/make_architecture_circular.py
"""
import json, math, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from make_tree import DOMAINS, prune  # noqa: E402
from make_architecture import spans_for_all  # noqa: E402

COL = dict(DOMAINS)
R_TREE, R_BAR, BAR_MAX = 230.0, 255.0, 230.0
R_LABEL = R_BAR + BAR_MAX + 18


def xy(r, a):
    rad = math.radians(a)
    return r * math.cos(rad), r * math.sin(rad)


def main():
    data = spans_for_all()
    t = json.load(open(ROOT / "tree_layout.json"))
    nodes = prune(t["nodes"], set(data))
    tax = {x["sp"]: x for x in t["leaf_taxonomies"]}
    rmax = max(n["r"] for n in nodes.values())
    for n in nodes.values():
        n["R"] = n["r"] / rmax * R_TREE
    maxlen = max(d["length"] for d in data.values())
    scale = BAR_MAX / maxlen

    fig = plt.figure(figsize=(16, 16), dpi=170, facecolor="white")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], aspect="equal")
    L = R_LABEL + 170
    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.axis("off")

    for n in nodes.values():
        if n["p"] is None:
            continue
        p = nodes[n["p"]]
        x0, y0 = xy(p["R"], n["a"]); x1, y1 = xy(n["R"], n["a"])
        ax.plot([x0, x1], [y0, y1], color="black", lw=0.7)
    for n in nodes.values():
        if n["leaf"] or not n["c"]:
            continue
        angs = [nodes[c]["a"] for c in n["c"]]
        th = [min(angs) + (max(angs) - min(angs)) * i / 60 for i in range(61)]
        pts = [xy(n["R"], a) for a in th]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color="black", lw=0.7)

    for n in nodes.values():
        if not n["leaf"]:
            continue
        sp, a = n["sp"], n["a"]
        d = data[sp]; tx = tax.get(sp, {})
        # supergroup marker
        mx, my = xy(R_BAR - 14, a)
        ax.scatter([mx], [my], s=22, color=tx.get("supergroup_color", "#999999"), zorder=4, lw=0)
        # grey protein length line
        x0, y0 = xy(R_BAR, a); x1, y1 = xy(R_BAR + d["length"] * scale, a)
        ax.plot([x0, x1], [y0, y1], color="#cbd5e1", lw=1.0, zorder=1)
        for key, (s, e) in d["spans"].items():
            p0 = xy(R_BAR + (s - 1) * scale, a); p1 = xy(R_BAR + e * scale, a)
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=COL[key], lw=5.0, solid_capstyle="butt", zorder=2)
        # taxon label, flipped on the left half
        lx, ly = xy(R_LABEL, a)
        rot = a if math.cos(math.radians(a)) >= 0 else a + 180
        ha = "left" if math.cos(math.radians(a)) >= 0 else "right"
        ax.text(lx, ly, sp.replace("_", " ") + f"  {d['length']} aa", rotation=rot, rotation_mode="anchor",
                ha=ha, va="center", fontsize=6.2, style="italic")

    ax.legend(handles=[Patch(color=c, label=d) for d, c in DOMAINS], loc="lower right",
              bbox_to_anchor=(1.0, 0.0), frameon=False, fontsize=11, title="SCAPER domain")
    sg = {}
    for sp in data:
        x = tax.get(sp, {})
        sg.setdefault(x.get("supergroup", "unknown"), x.get("supergroup_color", "#999999"))
    fig.legend(handles=[Patch(color=c, label=g) for g, c in sorted(sg.items())], loc="lower left",
               frameon=False, fontsize=7.5, title="Supergroup", ncol=2)
    fig.text(0.5, 0.975, "TCS tree: SCAPER protein architecture per species (orthogroup)",
             ha="center", fontsize=15, weight="bold")
    out = HERE / "scaper_orthogroup_architecture_circular.png"
    fig.savefig(out, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
