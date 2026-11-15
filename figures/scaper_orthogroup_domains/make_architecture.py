"""Rectangular TCS tree (SCAPER orthogroup species only) with each species'
SCAPER protein drawn to scale: grey bar = protein length, coloured boxes =
domains at their residue positions. Taxa are labelled and coloured by supergroup.

Domain spans (protein residue numbers):
  N-domain / zinc finger  hmmsearch envelope (PF16501 / PF12874, E <= 1e-3)
  alpha-helical / CTD     residues aligned to human 413-769 / 857-1395,
                          presence when >= 30% of the human span is covered

  python3 figures/scaper_orthogroup_domains/make_architecture.py
"""
import json, sys, tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "pipeline/helpers"))
sys.path.insert(0, str(HERE))
import build_scaper_regions as br  # noqa: E402
from make_tree import DOMAINS, COVER, prune  # noqa: E402

COL = dict(DOMAINS)


def spans_for_all():
    rows, human = br.load_rows()
    hcols = br.human_columns(human["s"])
    out = {}
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        info = {}
        for label, acc, key in [("N", "PF16501", "N-domain"), ("ZnF", "PF12874", "Zinc finger")]:
            _, inf, _ = br.profile_region(label, acc, rows, work)
            for name, (f, t) in inf.items():
                info.setdefault(name, {})[key] = (f, t)
    for r in rows:
        sp, name = r["sp"], f"{r['sp']}|{r['id']}"
        length = len(r["s"].replace("-", ""))
        spans = dict(info.get(name, {}))
        for key, (s, e) in [("α-helical", (413, 769)), ("α-solenoid CTD", (857, 1395))]:
            c0, c1 = hcols[s], hcols[e]
            cols = [i for i in range(c0, c1 + 1) if human["s"][i] != "-" and r["s"][i] != "-"]
            if len(cols) / (e - s + 1) >= COVER:
                # residue numbers of first and last aligned residue inside the slice
                first = r["s"][:cols[0]].replace("-", "")
                last = r["s"][:cols[-1] + 1].replace("-", "")
                spans[key] = (len(first) + 1, len(last))
        out[sp] = {"length": length, "spans": spans}
    return out


def main():
    data = spans_for_all()
    t = json.load(open(ROOT / "tree_layout.json"))
    nodes = prune(t["nodes"], set(data))
    tax = {x["sp"]: x for x in t["leaf_taxonomies"]}

    leaves = sorted((n for n in nodes.values() if n["leaf"]), key=lambda n: -n["a"])
    y = {}
    for i, n in enumerate(leaves):
        y[n["id"]] = len(leaves) - 1 - i          # top leaf = highest y
    for nid in sorted(nodes, reverse=True):      # children first
        n = nodes[nid]
        if not n["leaf"]:
            y[nid] = sum(y[c] for c in n["c"]) / len(n["c"])

    H = len(leaves)
    fig = plt.figure(figsize=(16, 0.16 * H + 2), dpi=150, facecolor="white")
    axT = fig.add_axes([0.02, 0.02, 0.30, 0.94])
    axL = fig.add_axes([0.33, 0.02, 0.17, 0.94])
    axA = fig.add_axes([0.52, 0.02, 0.46, 0.94])
    for ax in (axT, axL, axA):
        ax.set_ylim(-0.7, H - 0.3); ax.axis("off")
    axT.set_xlim(0, max(n["r"] for n in nodes.values()) * 1.02)
    for n in nodes.values():
        if n["p"] is None:
            continue
        p = nodes[n["p"]]
        axT.plot([p["r"], n["r"]], [y[n["id"]], y[n["id"]]], color="black", lw=0.7)
    for n in nodes.values():
        if n["leaf"]:
            continue
        ys = [y[c] for c in n["c"]]
        axT.plot([n["r"], n["r"]], [min(ys), max(ys)], color="black", lw=0.7)

    maxlen = max(d["length"] for d in data.values())
    axA.set_xlim(-20, maxlen * 1.02)
    for n in leaves:
        sp = n["sp"]; yy = y[n["id"]]; d = data[sp]; tx = tax.get(sp, {})
        axL.add_patch(Rectangle((0.02, yy - 0.42), 0.035, 0.84, transform=axL.transData,
                                color=tx.get("supergroup_color", "#999999")))
        axL.text(0.08, yy, sp.replace("_", " "), va="center", ha="left", fontsize=7.2, style="italic")
        axA.plot([0, d["length"]], [yy, yy], color="#cbd5e1", lw=1.0, zorder=1)
        for key, (s, e) in d["spans"].items():
            axA.add_patch(Rectangle((s, yy - 0.36), e - s + 1, 0.72, color=COL[key], zorder=2, lw=0))
        axA.text(d["length"] + 12, yy, f"{d['length']} aa", va="center", fontsize=6.5, color="#475569")
    axA.set_xlabel("protein residue", fontsize=10)
    # clean residue axis bottom
    axA.plot([0, maxlen], [-0.6, -0.6], color="black", lw=0.6)
    axA.set_ylim(-0.7, H - 0.3)
    fig.legend(handles=[Patch(color=c, label=d) for d, c in DOMAINS], loc="lower right", frameon=False,
               fontsize=11, title="SCAPER domain")
    sg = {}
    for sp in data:
        g = tax.get(sp, {}).get("supergroup", "unknown")
        sg.setdefault(g, tax.get(sp, {}).get("supergroup_color", "#999999"))
    fig.legend(handles=[Patch(color=c, label=g) for g, c in sorted(sg.items())], loc="lower left",
               frameon=False, fontsize=8, title="Supergroup (taxa)", bbox_to_anchor=(0.0, 0.0), ncol=2)
    fig.text(0.5, 0.985, "TCS tree: SCAPER protein architecture per species (orthogroup)",
             ha="center", fontsize=15, weight="bold")
    out = HERE / "scaper_orthogroup_architecture.png"
    fig.savefig(out, facecolor="white")
    print("wrote", out, "species", H)


if __name__ == "__main__":
    main()
