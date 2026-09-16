import os
import re
import json
import math
import struct
import pickle
import numpy as np
from ete3 import Tree

TCS_TREE_PATH = "/home/prodromosp/scaper_new/TCS.tree"
EUKPROT_PATH = "/home/prodromosp/scaper_new/junk/EukProt_included_data_sets.v03.2021_11_22.txt"
ORTHOGROUPS_PATH = "/home/prodromosp/scaper_new/orthogroups.pkl"

OUT_TREE_JSON = "/home/prodromosp/dollo-network-explorer/tree_layout.json"
OUT_PRESENCE_BIN = "/home/prodromosp/dollo-network-explorer/tree_presence.bin"
OUT_CACHE_TAX = "/home/prodromosp/dollo-network-explorer/cache/tree_taxonomy.json"


def main():
    print("1. Loading tree, EukProt metadata, and orthogroups...")
    with open(TCS_TREE_PATH, "r", encoding="utf-8") as f:
        tree_text = f.read()

    tree_match = re.search(r"tree\s+\S+\s*=\s*(\[&R\])?\s*(\(.*?\);)", tree_text, re.S)
    if not tree_match:
        raise ValueError("Could not find tree block in TCS.tree")
    tree = Tree(tree_match.group(2), format=1)
    outgroup_cand = [l for l in tree.get_leaves() if "Gefionella_okellyi" in l.name]
    if outgroup_cand:
        tree.set_outgroup(outgroup_cand[0])

    # Parse TCS taxlabels
    taxlabels_m = re.search(r"taxlabels\s+(.*?);", tree_text, re.S)
    tcs_parsed = {}
    if taxlabels_m:
        for line in taxlabels_m.group(1).strip().splitlines():
            line = line.strip()
            if not line:
                continue
            m = re.match(
                r"^([^@#|\[]+)(?:@([^#|\[]+))?(?:#([^|\[]+)\|([^|\[]+)\|([^|\[]+))?.*?(?:\[&!color=([^\]]+)\])?",
                line,
            )
            if m:
                sp = m.group(1).strip()
                clade = m.group(2).strip() if m.group(2) else ""
                stats = m.group(3).strip() if m.group(3) else ""
                cilia = m.group(4).strip() if m.group(4) else ""
                color = m.group(6).strip() if m.group(6) else ""
                tcs_parsed[sp] = {"clade": clade, "stats": stats, "cilia": cilia, "color": color}

    # Load EukProt dataset
    euk = {}
    with open(EUKPROT_PATH, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > 1:
                euk[parts[1]] = dict(zip(header, parts))

    # Load orthogroups
    with open(ORTHOGROUPS_PATH, "rb") as f:
        df_ortho = pickle.load(f)

    leaves = tree.get_leaves()
    n_leaves = len(leaves)
    print(f"  {n_leaves} leaves in tree.")

    leaf_species = []
    for l in leaves:
        m = re.match(r"^([^@#|\[]+)", l.name)
        leaf_species.append(m.group(1).strip() if m else l.name)

    for sp in leaf_species:
        if sp not in df_ortho.columns:
            raise ValueError(f"Species {sp} not found in orthogroups columns!")
        if sp not in euk:
            raise ValueError(f"Species {sp} not found in EukProt!")

    # Circular angles
    GAP_DEG = 0.0
    TOTAL_SPAN_DEG = 360.0
    START_DEG = 90.0

    leaf_angles = []
    for i, leaf in enumerate(leaves):
        ang = START_DEG - (i / n_leaves) * TOTAL_SPAN_DEG
        leaf.add_feature("_ang", ang)
        leaf_angles.append(round(ang, 3))

    for node in tree.traverse("postorder"):
        if not node.is_leaf():
            child_angs = [c._ang for c in node.children]
            node.add_feature("_ang", sum(child_angs) / len(child_angs))

    # Radii calculation
    R_ROOT = 35.0
    R_TREE = 300.0

    # 1. Molecular clock phylogram radius
    tree.add_feature("_dist_root", 0.0)
    for node in tree.traverse():
        if node.up is not None:
            node.add_feature(
                "_dist_root",
                node.up._dist_root + (node.dist if node.dist > 0 else 0.05),
            )

    max_dist = max(n._dist_root for n in tree.traverse())
    for node in tree.traverse():
        if node.is_leaf():
            node.add_feature("_r_phylo", R_TREE)
        else:
            node.add_feature(
                "_r_phylo",
                R_ROOT + (node._dist_root / max_dist) * (R_TREE - R_ROOT) * 0.95,
            )

    # 2. Cladogram radius (bottom-up height calculation)
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            node.add_feature("_height", 0)
        else:
            node.add_feature("_height", max(c._height for c in node.children) + 1)

    max_height = tree._height
    for node in tree.traverse():
        if node.is_leaf():
            node.add_feature("_r", R_TREE)
        else:
            node.add_feature(
                "_r",
                R_ROOT + ((max_height - node._height) / max_height) * (R_TREE - R_ROOT) * 0.95,
            )

    # 2. Clean UniProt / UniEuk Taxonomic Hierarchy
    print("2. Classifying taxonomy per leaf species...")

    supergroup_colors = {
        "Opisthokonta": "#3b82f6",     # Blue
        "SAR": "#ef4444",              # Red
        "Archaeplastida": "#22c55e",   # Emerald Green
        "Amoebozoa": "#f97316",        # Orange
        "Discoba": "#a855f7",          # Purple
        "Cryptista": "#ec4899",        # Pink
        "Haptista": "#b45309",         # Amber/Brown
        "CRuMs": "#14b8a6",            # Teal
        "Metamonada": "#8b5cf6",       # Violet
        "Apusomonadida": "#06b6d4",    # Cyan
        "Centroplasthelida": "#eab308",# Yellow
        "Malawimonadida": "#84cc16",   # Lime
        "Breviatea": "#6366f1",        # Indigo
        "Ancyromonadida": "#10b981",   # Green-cyan
        "Hemimastigophora": "#d946ef", # Magenta
        "Telonemia": "#64748b",        # Slate
        "Other Eukaryota": "#94a3b8",
    }

    kingdom_colors = {
        "Metazoa": "#2563eb",          # Vivid Blue
        "Choanoflagellata": "#38bdf8", # Sky Blue
        "Ichthyosporea": "#67e8f9",    # Light Cyan
        "Fungi": "#7c3aed",            # Deep Violet
        "Cristidiscoidea": "#c084fc",  # Lavender
        "Viridiplantae": "#16a34a",    # Green
        "Rhodophyta": "#dc2626",       # Crimson Red
        "Glaucophyta": "#0284c7",      # Deep Sky
        "Alveolata": "#9333ea",        # Purple
        "Stramenopiles": "#ea580c",    # Orange Red
        "Rhizaria": "#c026d3",         # Magenta
        "Amoebozoa": "#f59e0b",        # Amber
        "Euglenozoa": "#15803d",       # Dark Green
        "Heterolobosea": "#84cc16",    # Lime
        "Jakobida": "#059669",         # Sea Green
        "Fornicata": "#64748b",        # Slate
        "Parabasalia": "#475569",      # Dark Slate
        "Preaxostyla": "#78716c",      # Warm Gray
        "Haptophyta": "#b45309",       # Brown
        "Cryptophyceae": "#0d9488",    # Teal
        "Apusomonadida": "#06b6d4",    # Cyan
        "Centroplasthelida": "#eab308",# Yellow
        "Breviatea": "#6366f1",        # Indigo
        "Ancyromonadida": "#10b981",   # Green
        "Hemimastigophora": "#d946ef", # Pink-magenta
        "Telonemia": "#94a3b8",        # Silver
        "Other": "#94a3b8",
    }

    phylum_colors = {
        # Animals (Metazoa)
        "Chordata": "#1d4ed8",
        "Arthropoda": "#2563eb",
        "Nematoda": "#60a5fa",
        "Cnidaria": "#93c5fd",
        "Porifera": "#0284c7",
        "Placozoa": "#38bdf8",
        "Annelida": "#0369a1",
        "Echinodermata": "#0ea5e9",
        "Ctenophora": "#7dd3fc",
        # Unicellular Holozoa
        "Choanoflagellata": "#06b6d4",
        "Ichthyosporea": "#22d3ee",
        "Pluriformea": "#67e8f9",
        "Filasterea": "#a5f3fc",
        # Fungi
        "Ascomycota": "#581c87",
        "Basidiomycota": "#6b21a8",
        "Mucoromycota": "#7e22ce",
        "Glomeromycota": "#9333ea",
        "Zoopagomycota": "#a855f7",
        "Blastocladiomycota": "#c084fc",
        "Chytridiomycota": "#d8b4fe",
        "Aphelidiomycota": "#e9d5ff",
        "Rozellomycota": "#f3e8ff",
        "Rotosphaerida": "#8b5cf6",
        # Plants & Algae
        "Streptophyta": "#15803d",
        "Chlorophyta": "#22c55e",
        "Prasinodermophyta": "#4ade80",
        "Rhodophyta": "#dc2626",
        "Glaucophyta": "#0284c7",
        # Alveolata
        "Ciliophora": "#7c3aed",
        "Apicomplexa": "#9333ea",
        "Dinoflagellata": "#a855f7",
        "Perkinsozoa": "#c084fc",
        "Colponemidia": "#d8b4fe",
        # Stramenopiles
        "Oomycota": "#b45309",
        "Ochrophyta": "#d97706",
        "Opalozoa": "#f59e0b",
        "Bicosoecida": "#fbbf24",
        "Labyrinthulomycetes": "#fcd34d",
        # Rhizaria
        "Cercozoa": "#c026d3",
        "Foraminifera": "#d946ef",
        "Endomyxa": "#e879f9",
        # Amoebozoa
        "Tubulinea": "#ea580c",
        "Evosea": "#f97316",
        "Discosea": "#fb923c",
        # Discoba
        "Kinetoplastea": "#166534",
        "Diplonemea": "#15803d",
        "Euglenida": "#16a34a",
        "Heterolobosea": "#65a30d",
        "Jakobida": "#84cc16",
        # Others
        "Haptophyta": "#9a3412",
        "Cryptomonadales": "#0f766e",
        "Goniomonas": "#14b8a6",
        "Other": "#94a3b8",
    }

    tcs_fallback_colors = {
        "Alveolata": "#dcbeff",
        "Amoebozoa": "#f58231",
        "Anaeramoebidae": "#d73027",
        "Ancoracysta": "#ae017e",
        "Ancyromonadida": "#a1d99b",
        "Apusomonadida": "#aaffc3",
        "Barthelona": "#cab2d6",
        "Breviatea": "#984ea3",
        "Centroplasthelida": "#ffd92f",
        "Chloroplastida": "#9A6324",
        "Collodictyonidae": "#66c2a5",
        "Cryptophyceae": "#ffd8b1",
        "Euglenozoa": "#808000",
        "Fornicata": "#a9a9a9",
        "Glaucophyta": "#fabed4",
        "Haptophyta": "#800000",
        "Hemimastigophora": "#f768a1",
        "Heterolobosea": "#aaffc3",
        "Jakobida": "#3cb44b",
        "Malawimonadida": "#a6d854",
        "Mantamonas": "#8da0cb",
        "Opisthokonta": "#3b82f6",
        "Palpitomonas": "#2b8cbe",
        "Parabasalia": "#469990",
        "Preaxostyla": "#cccccc",
        "Rhizaria": "#911eb4",
        "Rhodelphidia": "#386cb0",
        "Rhodophyta": "#e6194B",
        "Rigifilida": "#a9a9a9",
        "Stramenopiles": "#f032e6",
        "Telonemia": "#bebada",
    }

    leaf_taxonomies = []
    tree_taxonomy_cache = {}

    for i, sp in enumerate(leaf_species):
        einfo = euk[sp]
        raw_tokens = [
            t.strip().strip("'") for t in einfo["Taxonomy_UniEuk"].split(";") if t.strip()
        ]
        tokens_low = [t.lower() for t in raw_tokens]
        sg_euk = einfo["Supergroup_UniEuk"]
        tg1 = einfo["Taxogroup1_UniEuk"]
        tg2 = einfo["Taxogroup2_UniEuk"]
        genus = einfo["Genus_UniEuk"]
        epithet = einfo["Epithet_UniEuk"].replace('"', "").strip()
        sci_name = f"{genus} {epithet}" if genus and epithet else sp.replace("_", " ")

        tcs_entry = tcs_parsed.get(sp, {})
        tcs_c = tcs_entry.get("clade") or sg_euk
        tcs_col = (
            tcs_entry.get("color")
            or tcs_fallback_colors.get(tcs_c, "#94a3b8")
        )
        if tcs_col == "#000000" and tcs_c in tcs_fallback_colors:
            tcs_col = tcs_fallback_colors[tcs_c]

        # 1. Supergroup
        if "opisthokonta" in tokens_low or sg_euk == "Opisthokonta":
            sg = "Opisthokonta"
        elif any(x in tokens_low for x in ["sar", "stramenopiles", "alveolata", "rhizaria"]) or sg_euk in ["Stramenopiles", "Alveolata", "Rhizaria"]:
            sg = "SAR"
        elif any(x in tokens_low for x in ["archaeplastida", "chloroplastida", "rhodophyta", "glaucophyta"]) or sg_euk in ["Chloroplastida", "Rhodophyta", "Glaucophyta", "Rhodelphidia"]:
            sg = "Archaeplastida"
        elif "amoebozoa" in tokens_low or sg_euk == "Amoebozoa":
            sg = "Amoebozoa"
        elif "discoba" in tokens_low or sg_euk in ["Euglenozoa", "Heterolobosea", "Jakobida"]:
            sg = "Discoba"
        elif "crums" in tokens_low or sg_euk in ["Collodictyonidae", "Rigifilida", "Mantamonas"]:
            sg = "CRuMs"
        elif "metamonada" in tokens_low or sg_euk in ["Fornicata", "Parabasalia", "Preaxostyla", "Barthelona"]:
            sg = "Metamonada"
        elif "cryptista" in tokens_low or sg_euk == "Cryptophyceae":
            sg = "Cryptista"
        elif "haptista" in tokens_low or sg_euk == "Haptophyta":
            sg = "Haptista"
        else:
            sg = sg_euk

        # 2. Kingdom
        if sg == "Opisthokonta":
            if "metazoa" in tokens_low or tg1 == "Metazoa":
                kg = "Metazoa"
            elif any(x in tokens_low for x in ["choanoflagellata", "choanozoa"]) or tg1 == "Choanoflagellata":
                kg = "Choanoflagellata"
            elif any(x in tokens_low for x in ["ichthyosporea", "pluriformea", "filasterea"]) or tg1 in ["Ichthyosporea", "Pluriformea", "Filasterea"]:
                kg = "Ichthyosporea"
            elif "fungi" in tokens_low or tg1 == "Fungi":
                kg = "Fungi"
            elif any(x in tokens_low for x in ["rotosphaerida", "cristidiscoidea", "nuclearia", "fonticula"]) or tg1 == "Rotosphaerida":
                kg = "Cristidiscoidea"
            else:
                kg = tg1
        elif sg == "SAR":
            if "stramenopiles" in tokens_low or sg_euk == "Stramenopiles":
                kg = "Stramenopiles"
            elif "alveolata" in tokens_low or sg_euk == "Alveolata":
                kg = "Alveolata"
            elif "rhizaria" in tokens_low or sg_euk == "Rhizaria":
                kg = "Rhizaria"
            else:
                kg = "SAR"
        elif sg == "Archaeplastida":
            if any(x in tokens_low for x in ["chloroplastida", "viridiplantae"]) or sg_euk == "Chloroplastida":
                kg = "Viridiplantae"
            elif any(x in tokens_low for x in ["rhodophyta", "rhodelphidia"]) or sg_euk in ["Rhodophyta", "Rhodelphidia"]:
                kg = "Rhodophyta"
            elif "glaucophyta" in tokens_low or sg_euk == "Glaucophyta":
                kg = "Glaucophyta"
            else:
                kg = "Archaeplastida"
        elif sg == "Discoba":
            if "euglenozoa" in tokens_low or sg_euk == "Euglenozoa":
                kg = "Euglenozoa"
            elif "heterolobosea" in tokens_low or sg_euk == "Heterolobosea":
                kg = "Heterolobosea"
            elif "jakobida" in tokens_low or sg_euk == "Jakobida":
                kg = "Jakobida"
            else:
                kg = "Discoba"
        elif sg == "Metamonada":
            if "fornicata" in tokens_low or sg_euk == "Fornicata":
                kg = "Fornicata"
            elif "parabasalia" in tokens_low or sg_euk == "Parabasalia":
                kg = "Parabasalia"
            elif "preaxostyla" in tokens_low or sg_euk == "Preaxostyla":
                kg = "Preaxostyla"
            else:
                kg = sg_euk
        elif sg == "CRuMs":
            kg = sg_euk
        else:
            kg = sg_euk

        # 3. Phylum
        if kg == "Metazoa":
            for p in ["Chordata", "Arthropoda", "Nematoda", "Cnidaria", "Porifera", "Placozoa", "Annelida", "Echinodermata", "Ctenophora"]:
                if p.lower() in tokens_low or tg2.lower() == p.lower():
                    ph = p
                    break
            else:
                ph = tg2 if tg2 else "Metazoa"
        elif kg == "Fungi":
            for p in ["Ascomycota", "Basidiomycota", "Mucoromycota", "Glomeromycota", "Zoopagomycota", "Blastocladiomycota", "Chytridiomycota", "Aphelidiomycota", "Rozellomycota"]:
                if p.lower() in tokens_low or tg2.lower() == p.lower():
                    ph = p
                    break
            else:
                if "aphelidea" in tokens_low or tg2 == "Aphelidea":
                    ph = "Aphelidiomycota"
                elif "rozellida" in tokens_low or tg2 == "Rozellida":
                    ph = "Rozellomycota"
                else:
                    ph = tg2 if tg2 else "Fungi"
        elif kg == "Viridiplantae":
            if "streptophyta" in tokens_low or tg1 == "Streptophyta":
                ph = "Streptophyta"
            elif "palmophyllophyceae" in tokens_low or tg1 == "Palmophyllophyceae":
                ph = "Prasinodermophyta"
            else:
                ph = "Chlorophyta"
        elif kg == "Alveolata":
            if "ciliophora" in tokens_low or tg1 == "Ciliophora":
                ph = "Ciliophora"
            elif "apicomplexa" in tokens_low or tg1 == "Apicomplexa":
                ph = "Apicomplexa"
            elif "dinoflagellata" in tokens_low or tg1 == "Dinoflagellata":
                ph = "Dinoflagellata"
            elif "perkinsea" in tokens_low or tg1 == "Perkinsea":
                ph = "Perkinsozoa"
            elif "colponemids" in tokens_low or tg1 == "colponemids":
                ph = "Colponemidia"
            else:
                ph = tg1
        elif kg == "Stramenopiles":
            if "oomycota" in tokens_low or "peronosporomycetes" in tokens_low:
                ph = "Oomycota"
            elif any(x in tokens_low for x in ["ochrophyta", "phaeophyceae", "bacillariophyta"]):
                ph = "Ochrophyta"
            elif "bicosoecida" in tokens_low or tg2 == "Bicosoecida":
                ph = "Bicosoecida"
            elif "sagenista" in tokens_low or tg1 == "Sagenista":
                ph = "Labyrinthulomycetes"
            elif "opalozoa" in tokens_low or tg1 == "Opalozoa":
                ph = "Opalozoa"
            else:
                ph = tg1
        elif kg == "Rhizaria":
            if "cercozoa" in tokens_low or tg1 == "Cercozoa":
                ph = "Cercozoa"
            elif "foraminifera" in tokens_low or tg1 == "Foraminifera":
                ph = "Foraminifera"
            elif "endomyxa" in tokens_low or tg1 == "Endomyxa":
                ph = "Endomyxa"
            else:
                ph = tg1
        elif kg == "Amoebozoa":
            if any(x in tokens_low for x in ["tubulinea", "euamoebida"]):
                ph = "Tubulinea"
            elif any(x in tokens_low for x in ["evosea", "variosea", "eumycetozoa"]):
                ph = "Evosea"
            elif "discosea" in tokens_low or tg1 == "Discosea":
                ph = "Discosea"
            else:
                ph = tg1
        elif kg == "Euglenozoa":
            if "kinetoplastea" in tokens_low or tg1 == "Kinetoplastea":
                ph = "Kinetoplastea"
            elif "diplonemea" in tokens_low or tg1 == "Diplonemea":
                ph = "Diplonemea"
            elif "euglenida" in tokens_low or tg1 == "Euglenida":
                ph = "Euglenida"
            else:
                ph = "Euglenozoa"
        elif tg1 in ["Choanoflagellata", "Ichthyosporea", "Filasterea", "Pluriformea", "Rotosphaerida"]:
            ph = tg1
        else:
            ph = tg1 if tg1 else kg

        # Palette colors with consistent fallback
        sg_col = supergroup_colors.get(sg, "#94a3b8")
        kg_col = kingdom_colors.get(kg, tcs_fallback_colors.get(kg, "#94a3b8"))
        ph_col = phylum_colors.get(ph, kg_col)

        leaf_taxonomies.append({
            "sp": sp,
            "sci_name": sci_name,
            "supergroup": sg,
            "supergroup_color": sg_col,
            "kingdom": kg,
            "kingdom_color": kg_col,
            "phylum": ph,
            "phylum_color": ph_col,
            "detailed": ph,
            "detailed_color": ph_col,
            "tcs": tcs_c,
            "tcs_color": tcs_col,
            "cilia": tcs_entry.get("cilia", ""),
        })

        tree_taxonomy_cache[sp] = {
            "matched_query": sci_name,
            "scientificName": sci_name,
            "rank": "species",
            "supergroup": sg,
            "kingdom": kg,
            "phylum": ph,
            "lineage": raw_tokens,
            "raw_leaf": tcs_entry.get("raw_leaf", sp),
            "tree_clade": tcs_c,
            "tree_color": tcs_col,
        }

    # 3. Generate clade blocks for each level
    def build_blocks_for_level(level_name, color_key):
        blocks = []
        cur_block = None
        for i, item in enumerate(leaf_taxonomies):
            c_val = item[level_name]
            c_col = item[color_key]
            ang = leaf_angles[i]
            if cur_block is None:
                cur_block = {
                    "clade": c_val,
                    "color": c_col,
                    "start_idx": i,
                    "end_idx": i,
                    "start_angle": ang,
                    "end_angle": ang,
                    "species": [item["sp"]],
                }
            elif cur_block["clade"] == c_val:
                cur_block["end_idx"] = i
                cur_block["end_angle"] = ang
                cur_block["species"].append(item["sp"])
            else:
                blocks.append(cur_block)
                cur_block = {
                    "clade": c_val,
                    "color": c_col,
                    "start_idx": i,
                    "end_idx": i,
                    "start_angle": ang,
                    "end_angle": ang,
                    "species": [item["sp"]],
                }
        if cur_block:
            blocks.append(cur_block)
        return blocks

    clade_levels = {
        "kingdom": build_blocks_for_level("kingdom", "kingdom_color"),
        "phylum": build_blocks_for_level("phylum", "phylum_color"),
        "supergroup": build_blocks_for_level("supergroup", "supergroup_color"),
        "detailed": build_blocks_for_level("detailed", "detailed_color"),
        "tcs": build_blocks_for_level("tcs", "tcs_color"),
    }

    for lvl, blist in clade_levels.items():
        print(f"  Level '{lvl}': {len(blist)} contiguous sectors.")
        if lvl == "kingdom":
            for b in blist:
                if b["clade"] in ["Metazoa", "Fungi", "Choanoflagellata", "Ichthyosporea", "Cristidiscoidea"]:
                    print(f"    [Opisthokonta Kingdom] {b['clade']:20s} len={len(b['species']):2d} [{b['start_idx']}..{b['end_idx']}]")

    # 4. Node index and descendant leaves
    all_nodes = [n for n in tree.traverse()]
    node_to_id = {node: i for i, node in enumerate(all_nodes)}
    leaf_sp_to_idx = {sp: i for i, sp in enumerate(leaf_species)}

    nodes_data = []
    for node in all_nodes:
        nid = node_to_id[node]
        pid = node_to_id[node.up] if node.up is not None else None
        cids = [node_to_id[c] for c in node.children]

        leaf_names = [
            re.match(r"^([^@#|\[]+)", l.name).group(1).strip()
            for l in node.get_leaves()
        ]
        desc_indices = [leaf_sp_to_idx[l] for l in leaf_names]

        is_leaf = node.is_leaf()
        l_idx = leaf_sp_to_idx[leaf_names[0]] if is_leaf else None
        sp_name = leaf_species[l_idx] if is_leaf else None

        nodes_data.append({
            "id": nid,
            "p": pid,
            "c": cids,
            "r": round(node._r, 1),
            "r_phylo": round(node._r_phylo, 1),
            "a": round(node._ang, 3),
            "leaf": is_leaf,
            "leaf_idx": l_idx,
            "sp": sp_name,
            "leaves": desc_indices,
        })

    # 5. Pack presence matrix
    print("3. Packing presence binary matrix...")
    df_ordered = df_ortho[leaf_species].astype(np.uint8)
    gene_names = df_ordered.index.tolist()

    packed_bytes = bytearray()
    for gname in gene_names:
        row_bits = df_ordered.loc[gname].values
        b = bytearray(25)
        for i, val in enumerate(row_bits):
            if val:
                byte_idx = i // 8
                bit_idx = i % 8
                b[byte_idx] |= 1 << bit_idx
        packed_bytes.extend(b)

    header = struct.pack("<4sIH", b"DLTP", len(gene_names), n_leaves)
    with open(OUT_PRESENCE_BIN, "wb") as f_bin:
        f_bin.write(header)
        f_bin.write(packed_bytes)
    print(f"  Wrote presence binary: {OUT_PRESENCE_BIN} ({len(header) + len(packed_bytes)} bytes)")

    # 6. Save Tree Layout JSON
    print("4. Writing tree layout JSON...")
    out_dict = {
        "n_leaves": n_leaves,
        "leaf_species": leaf_species,
        "leaf_angles": leaf_angles,
        "leaf_taxonomies": leaf_taxonomies,
        "clade_levels": clade_levels,
        "clade_blocks": clade_levels["kingdom"],  # Default clean blocks
        "nodes": nodes_data,
        "gene_names": gene_names,
        "gap_deg": GAP_DEG,
        "start_deg": START_DEG,
        "r_root": R_ROOT,
        "r_tree": R_TREE,
    }

    with open(OUT_TREE_JSON, "w", encoding="utf-8") as f_out:
        json.dump(out_dict, f_out, separators=(",", ":"))
    print(f"  Wrote tree layout JSON: {OUT_TREE_JSON} ({os.path.getsize(OUT_TREE_JSON)} bytes)")

    # 7. Update cache/tree_taxonomy.json
    with open(OUT_CACHE_TAX, "w", encoding="utf-8") as f_tax:
        json.dump(tree_taxonomy_cache, f_tax, indent=2)
    print(f"  Updated cache: {OUT_CACHE_TAX}")
    print("\nTree construction finished successfully!")


if __name__ == "__main__":
    main()
