import sys
import pickle
from pathlib import Path

print("1. Loading human gene list...")
with open("/home/prodromosp/scaper_new/orthogroups.pkl", "rb") as f:
    target_genes = set(pickle.load(f).index)

print("2. Mapping human genes to PANTHER families...")
panther_human = "/home/prodromosp/panther_test/PTHR19.0_human"
gene_to_pthr = {}
with open(panther_human, "r") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 4:
            sym = parts[2]
            pthr = parts[3].split(":")[0] # e.g. PTHR23158
            if sym in target_genes:
                gene_to_pthr[sym] = pthr

print(f"Mapped {len(gene_to_pthr)} / {len(target_genes)} genes to PANTHER families.")
import json
with open("/home/prodromosp/panther_eggnog_pipeline/data/gene_to_pthr.json", "w") as f:
    json.dump(gene_to_pthr, f)
