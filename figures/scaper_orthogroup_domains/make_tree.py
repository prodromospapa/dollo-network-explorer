"""TCS 196-taxa tree with per-species SCAPER domain presence (SCAPER orthogroup).

Presence per species (sequence = best DIAMOND hit to human, from
alignments_orthogroup/SCAPER.json.gz):
  N-domain   PF16501 (SCAPER_N) hmmsearch hit, E <= 1e-3
  ZnF        PF12874 (zf-met) hmmsearch hit, E <= 1e-3
  helical    >= 30% of human 413-769 covered by the species' aligned residues
  CTD        >= 30% of human 857-1395 covered by the species' aligned residues

  python3 figures/scaper_orthogroup_domains/make_tree.py
"""
import gzip, json, math, sys, tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline/helpers"))
import build_scaper_regions as br  # noqa: E402

OUT = Path(__file__).resolve().parent
COVER = 0.30
DOMAINS = [  # ring order, outer label, colour (from the SCAPER domain diagram)
    ("N-domain", "#f08080"), ("α-helical", "#f59e2a"), ("Zinc finger", "#1e90ff"), ("α-solenoid CTD", "#e879b9")]


def presence_table():
    rows, human = br.load_rows()
    hcols = br.human_columns(human["s"])
    table = {r["sp"]: {} for r in rows}
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        for label, acc, key in [("N", "PF16501", "N-domain"), ("ZnF", "PF12874", "Zinc finger")]:
            _, info, _ = br.profile_region(label, acc, rows, work)
            for name in info:
                sp = name.split("|")[0]
                table[sp][key] = True
    for key, (s, e) in [("α-helical", (413, 769)), ("α-solenoid CTD", (857, 1395))]:
        c0, c1 = hcols[s], hcols[e]
        for r in rows:
            cover = sum(1 for i in range(c0, c1 + 1) if human["s"][i] != "-" and r["s"][i] != "-") \
                    / (e - s + 1)
            table[r["sp"]][key] = cover >= COVER
    return {sp: {d: bool(table[sp].get(d, False)) for d, _ in DOMAINS} for sp in table}


def prune(nodes_list, keep):
    """Keep only leaves whose species are in `keep`, collapse single-child nodes,
    keep each surviving node's radius, and re-space leaves evenly around the circle."""
    byid = {n["id"]: n for n in nodes_list}

    def build(nid):
        n = byid[nid]
        if n["leaf"]:
            return {"leaf": True, "sp": n["sp"], "r": n["r"], "kids": []} if n["sp"] in keep else None
        kids = [k for k in (build(c) for c in n["c"]) if k is not None]
        if not kids:
            return None
        if len(kids) == 1:
            return kids[0]
        return {"leaf": False, "sp": None, "r": n["r"], "kids": kids}

    root = build(nodes_list[0]["id"])
    out, order = {}, []

    def flatten(node, parent):
        nid = len(out)
        out[nid] = {"id": nid, "p": parent, "c": [], "r": node["r"], "leaf": node["leaf"],
                    "sp": node["sp"], "a": None}
        if parent is not None:
            out[parent]["c"].append(nid)
        if node["leaf"]:
            order.append(nid)
        for k in node["kids"]:
            flatten(k, nid)
        return nid

    flatten(root, None)
    L = len(order)
    for i, nid in enumerate(order):
        out[nid]["a"] = 90.0 - i * 360.0 / L
    for nid in sorted(out, reverse=True):   # children have higher ids than parents
        n = out[nid]
        if not n["leaf"]:
            n["a"] = sum(out[c]["a"] for c in n["c"]) / len(n["c"])
    return out


def main():
    pres = presence_table()
    with open(OUT / "presence.tsv", "w") as fh:
        fh.write("species\t" + "\t".join(d for d, _ in DOMAINS) + "\n")
        for sp in sorted(pres):
            fh.write(sp + "\t" + "\t".join("1" if pres[sp][d] else "0" for d, _ in DOMAINS) + "\n")
    print({d: sum(1 for sp in pres if pres[sp][d]) for d, _ in DOMAINS}, "species present (of", len(pres), "with a sequence)")

    t = json.load(open(ROOT / "tree_layout.json"))
    nodes = prune(t["nodes"], set(pres))
    print("tree leaves kept:", sum(1 for n in nodes.values() if n["leaf"]))
    fig = plt.figure(figsize=(13, 13), dpi=200, facecolor="white")
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96], aspect="equal")
    ax.set_xlim(-400, 400); ax.set_ylim(-400, 400); ax.axis("off")

    def xy(r, a):
        rad = math.radians(a)
        return r * math.cos(rad), r * math.sin(rad)

    for n in nodes.values():
        if n["p"] is None:
            continue
        p = nodes[n["p"]]
        x0, y0 = xy(p["r"], n["a"]); x1, y1 = xy(n["r"], n["a"])
        ax.plot([x0, x1], [y0, y1], color="black", lw=0.6)
    for n in nodes.values():
        if n["leaf"] or not n["c"]:
            continue
        angs = [nodes[c]["a"] for c in n["c"]]
        th = [min(angs) + (max(angs) - min(angs)) * i / 40 for i in range(41)]
        pts = [xy(n["r"], a) for a in th]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color="black", lw=0.6)

    # domain rings: one filled bar per leaf per domain
    ring_r0, ring_w, ring_gap = 300, 14, 3
    absent = "#e5e7eb"
    for n in nodes.values():
        if not n["leaf"]:
            continue
        sp = n["sp"]
        for k, (d, col) in enumerate(DOMAINS):
            r0 = ring_r0 + k * (ring_w + ring_gap)
            on = pres.get(sp, {}).get(d, False)
            x0, y0 = xy(r0, n["a"]); x1, y1 = xy(r0 + ring_w, n["a"])
            ax.plot([x0, x1], [y0, y1], color=col if on else absent, lw=1.2, solid_capstyle="butt")

    handles = [Patch(color=c, label=d) for d, c in DOMAINS]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=11, title="SCAPER domain present")
    fig.text(0.5, 0.975, "TCS tree: SCAPER domain presence (orthogroup)", ha="center", fontsize=16, weight="bold")
    fig.savefig(OUT / "scaper_orthogroup_domains_tree.png", facecolor="white")
    print("wrote", OUT / "scaper_orthogroup_domains_tree.png")


if __name__ == "__main__":
    main()
