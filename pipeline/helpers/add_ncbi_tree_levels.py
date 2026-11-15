#!/usr/bin/env python3
"""
Add the EggNOG taxonomy ranks (domain, kingdom, phylum, class, order, family,
genus) to the TCS species-tree layout, so the TCS tree's Taxonomy dropdown
offers the same levels as the EggNOG tree.

Reads tcs_taxonomy_ncbi.json (from export_tcs_ncbi_taxonomy.py) and updates
tree_layout.json in place:
  * leaf_taxonomies[i] gains "<rank>" and "<rank>_color" for every NCBI rank
    (existing curated fields such as tcs/supergroup are kept);
  * clade_levels["<rank>"] gets contiguous clade blocks in leaf order, with
    angles taken from leaf_angles, in the same schema as the existing levels.

Usage (repo root):
  python3 pipeline/helpers/add_ncbi_tree_levels.py [tree_layout.json]
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TAX = ROOT / "tcs_taxonomy_ncbi.json"
UNCLASSIFIED_COLOR = "#a9a9a9"
# Distinct, colour-blind-aware cycle; adjacent clades get different colours.
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf",
           "#bcbd22", "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173", "#5254a3", "#6b6ecf",
           "#b5cf6b", "#cedb9c", "#e7ba52", "#ad494a", "#d6616b", "#e7969c", "#a55194", "#ce6dbd"]


def main():
    layout_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tree_layout.json"
    backup = layout_path.with_suffix(".json.bak")
    if not backup.exists():
        shutil.copy(layout_path, backup)
    layout = json.loads(layout_path.read_text())
    tax = json.loads(TAX.read_text())
    ranks = tax["tax_levels"]
    species = tax["species"]
    leaves = layout["leaf_species"]
    angles = layout["leaf_angles"]

    # Colours: one per distinct taxon per rank, stable across runs (sorted names)
    colour = {}
    for rank in ranks:
        names = sorted({species[sp][rank] for sp in leaves} - {"Unclassified"})
        colour[rank] = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(names)}
        colour[rank]["Unclassified"] = UNCLASSIFIED_COLOR

    for i, sp in enumerate(leaves):
        info = layout["leaf_taxonomies"][i]
        for rank in ranks:
            name = species[sp][rank]
            info[rank] = name
            info[rank + "_color"] = colour[rank][name]

    clade_levels = layout.setdefault("clade_levels", {})
    for rank in ranks:
        blocks, start = [], 0
        for i in range(1, len(leaves) + 1):
            if i == len(leaves) or species[leaves[i]][rank] != species[leaves[start]][rank]:
                name = species[leaves[start]][rank]
                members = leaves[start:i]
                blocks.append({
                    "clade": name,
                    "color": colour[rank][name],
                    "start_idx": start,
                    "end_idx": i - 1,
                    "start_angle": angles[start],
                    "end_angle": angles[i - 1],
                    "species": members,
                })
                start = i
        clade_levels[rank] = blocks

    layout_path.write_text(json.dumps(layout))
    print(f"Updated {layout_path}: {len(ranks)} ranks, "
          + ", ".join(f"{r}={len(clade_levels[r])} blocks" for r in ranks))


if __name__ == "__main__":
    main()
