import gzip, json
from collections import defaultdict
import sys
with open("pipeline/full_eggnog_pipeline/data/gene_to_pthr.json") as f:
    gene_to_pthr = json.load(f)
protid_to_genes = defaultdict(list)
with open("data/eggnog_dataset/gene_to_ensp.json") as f:
    gene_to_ensp = json.load(f)
    for g, ensps in gene_to_ensp.items():
        if g in gene_to_pthr:
            for ensp in ensps:
                protid_to_genes[f"9606.{ensp}"].append(g)

count = 0
found = 0
with gzip.open("data/e7.protein_families.tsv.gz", 'rt') as f:
    for line in f:
        if line.startswith("#"): continue
        count += 1
        parts = line.strip().split('\t')
        if len(parts) < 6: continue
        proteins = parts[4].split(',')
        for p in proteins:
            if p in protid_to_genes:
                found += 1
                break
        if count >= 100000:
            break
print(f"Scanned {count}, found matches in {found} families")
