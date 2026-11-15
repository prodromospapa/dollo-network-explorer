"""Circular TCS tree (SCAPER orthogroup species) in the explorer's style:
tree with a supergroup tip ring, shaded and labelled monophyletic clades, and
each species' SCAPER protein drawn as radial bars (length to scale, domains
coloured at their residue positions).

  python3 figures/scaper_orthogroup_domains/make_explorer_style.py
"""
import json, math, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Wedge

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from make_tree import DOMAINS, prune  # noqa: E402
from make_architecture import spans_for_all  # noqa: E402
import pickle  # noqa: E402
sys.path.insert(0, str(ROOT / "pipeline/helpers"))
import build_tcs_gene_alignments as tcs  # noqa: E402

PROT2GENE = Path("/mnt/home2/prodromosp/scaper_new/protein_dict.pkl")


def ortholog_species():
    """Species with an OrthoFinder ortholog of human SCAPER (TCS ortholog dataset)."""
    prot2gene = pickle.load(open(PROT2GENE, "rb"))
    _human, cands = tcs.ortholog_candidates({"SCAPER"}, prot2gene)["SCAPER"]
    return set(cands)

COL = dict(DOMAINS)
R_TREE, R_RING, R_BAR, BAR_MAX = 230.0, 240.0, 262.0, 200.0
MIN_CLADE = 4


def xy(r, a):
    rad = math.radians(a)
    return r * math.cos(rad), r * math.sin(rad)


ABBR = {"N-domain": "N", "α-helical": "H", "Zinc finger": "Z", "α-solenoid CTD": "C"}
GROUP_COLS = ["#111111", "#facc15", "#d946ef", "#0f766e", "#78350f", "#fb7185", "#64748b", "#d6d3d1", "#1e3a8a"]


def architecture(spans):
    """Domain architecture label, e.g. 'N+H+Z+C' (present domains only)."""
    names = [ABBR[k] for k in ABBR if k in spans]
    return "+".join(names) if names else "none"


def main():
    data = spans_for_all()
    t = json.load(open(ROOT / "tree_layout.json"))
    nodes = prune(t["nodes"], set(data))
    tax = {x["sp"]: x for x in t["leaf_taxonomies"]}
    rmax = max(n["r"] for n in nodes.values())
    for n in nodes.values():
        n["R"] = n["r"] / rmax * R_TREE

    # leaves under each node, and their angles
    leaves_of = {}
    for nid in sorted(nodes, reverse=True):
        n = nodes[nid]
        leaves_of[nid] = [nid] if n["leaf"] else [x for c in n["c"] for x in leaves_of[c]]
    sp_of = {nid: nodes[nid]["sp"] for nid in nodes if nodes[nid]["leaf"]}
    group_of = {sp: tax.get(sp, {}).get("supergroup", "unknown") for sp in data}
    gcol = {}
    for sp in data:
        gcol.setdefault(group_of[sp], tax.get(sp, {}).get("supergroup_color", "#999999"))

    # maximal monophyletic clades per supergroup
    clades = []
    for nid, n in nodes.items():
        if n["leaf"]:
            continue
        ls = leaves_of[nid]
        groups = {group_of[sp_of[l]] for l in ls}
        if len(groups) != 1 or len(ls) < MIN_CLADE:
            continue
        parent = n["p"]
        if parent is not None:
            pl = leaves_of[parent]
            if len({group_of[sp_of[l]] for l in pl}) == 1:
                continue   # not maximal
        clades.append((groups.pop(), nid, ls))

    orth = ortholog_species()
    print("ortholog species:", len(orth), "in orthogroup:", len(orth & set(data)))
    maxlen = max(d["length"] for d in data.values())
    scale = BAR_MAX / maxlen
    fig = plt.figure(figsize=(16, 16), dpi=170, facecolor="white")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], aspect="equal")
    L = R_BAR + BAR_MAX + 30
    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.axis("off")

    half = 360.0 / max(1, len(sp_of)) / 2
    for g, nid, ls in clades:
        angs = [nodes[l]["a"] for l in ls]
        t0, t1 = min(angs) - half, max(angs) + half
        ax.add_patch(Wedge((0, 0), R_BAR + BAR_MAX + 10, t0, t1, width=R_BAR + BAR_MAX + 10 - R_TREE,
                           color=gcol[g], alpha=0.22, lw=0))
        mid = (t0 + t1) / 2
        if len(ls) >= 6:
            lx, ly = xy(R_TREE - 18, mid)
            rot = mid if math.cos(math.radians(mid)) >= 0 else mid + 180
            ax.text(lx, ly, g, rotation=rot, rotation_mode="anchor", ha="right", va="center",
                    fontsize=10, weight="bold", color=gcol[g])

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

    for nid, sp in sp_of.items():
        a = nodes[nid]["a"]
        x0, y0 = xy(R_RING, a); x1, y1 = xy(R_RING + 8, a)
        ax.plot([x0, x1], [y0, y1], color=gcol[group_of[sp]], lw=2.2, solid_capstyle="butt")
        d = data[sp]
        if sp in orth:
            mx, my = xy(R_BAR - 8, a)
            ax.scatter([mx], [my], s=16, color="black", zorder=5, lw=0)
        x0, y0 = xy(R_BAR, a); x1, y1 = xy(R_BAR + d["length"] * scale, a)
        ax.plot([x0, x1], [y0, y1], color="#cbd5e1", lw=1.0, zorder=1)
        for key, (s, e) in d["spans"].items():
            p0 = xy(R_BAR + (s - 1) * scale, a); p1 = xy(R_BAR + e * scale, a)
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=COL[key], lw=4.5, solid_capstyle="butt", zorder=2)

    ax.legend(handles=[Patch(color=c, label=d) for d, c in DOMAINS], loc="lower right",
              bbox_to_anchor=(1.0, 0.0), frameon=False, fontsize=11, title="SCAPER domain")
    # domain-architecture groups: outer ring + table
    arch = {sp: architecture(data[sp]["spans"]) for sp in data}
    order = sorted(set(arch.values()), key=lambda a: (-sum(1 for x in arch.values() if x == a), a))
    acol = {a: GROUP_COLS[i % len(GROUP_COLS)] for i, a in enumerate(order)}
    for nid, sp in sp_of.items():
        a = nodes[nid]["a"]
        x0, y0 = xy(R_BAR + BAR_MAX + 14, a); x1, y1 = xy(R_BAR + BAR_MAX + 30, a)
        ax.plot([x0, x1], [y0, y1], color=acol[arch[sp]], lw=6.0, solid_capstyle="butt")
    with open(HERE / "domain_groups.tsv", "w") as fh:
        fh.write("architecture\tcount\tortholog_of_human_SCAPER\tspecies\n")
        for a in order:
            members = sorted(sp for sp in data if arch[sp] == a)
            fh.write(f"{a}\t{len(members)}\t{sum(1 for m in members if m in orth)}\t{', '.join(m.replace('_', ' ') for m in members)}\n")
    print("architecture groups:", {a: sum(1 for x in arch.values() if x == a) for a in order})
    handles = [Patch(color=acol[a], label=f"{a} ({sum(1 for x in arch.values() if x == a)})") for a in order]
    handles.append(Patch(color="white", label=""))
    handles += [Patch(color=c, label=g) for g, c in sorted(gcol.items())]
    handles.append(plt.Line2D([], [], marker="o", ls="", color="black", markersize=5,
                              label="ortholog of human SCAPER (OrthoFinder)"))
    fig.legend(handles=handles, loc="lower left", frameon=False, fontsize=7.5, ncol=2)
    fig.text(0.5, 0.975, "TCS tree: SCAPER protein architecture (orthogroup)", ha="center",
             fontsize=15, weight="bold")
    out = HERE / "scaper_orthogroup_explorer_style.png"
    fig.savefig(out, facecolor="white")
    print("wrote", out, "clades", len(clades))


if __name__ == "__main__":
    main()
