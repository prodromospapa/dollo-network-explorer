import gzip
import json
import os
from collections import defaultdict
from ete3 import Tree
import sys

print("Loading gene mapping...")
with open("pipeline/full_eggnog_pipeline/data/gene_to_pthr.json") as f:
    gene_to_pthr = json.load(f)

print("Loading target species from full EggNOG tree...")
tree = Tree("data/eggnog_dataset/eggnog_tree.newick", format=1)
target_taxids = set(leaf.name for leaf in tree.get_leaves())
print(f"Loaded {len(target_taxids)} target species.")

protid_to_genes = defaultdict(list)
with open("data/eggnog_dataset/gene_to_ensp.json") as f:
    gene_to_ensp = json.load(f)
    for g, ensps in gene_to_ensp.items():
        if g in gene_to_pthr:
            for ensp in ensps:
                protid_to_genes[f"9606.{ensp}"].append(g)

print(f"Targeting {len(protid_to_genes)} human proteins.")

gene_to_target_proteins = defaultdict(set)
path = "data/e7.protein_families.tsv.gz"

print("Streaming EggNOG families...", flush=True)
count = 0
with gzip.open(path, 'rt') as f:
    for line in f:
        if line.startswith("#"): continue
        count += 1
        if count % 100000 == 0:
            print(f"Processed {count} families...", flush=True)
        
        parts = line.strip().split('\t')
        if len(parts) < 6: continue
        proteins = parts[4].split(',')
        
        matched_genes = set()
        for p in proteins:
            if p in protid_to_genes:
                for g in protid_to_genes[p]:
                    matched_genes.add(g)
        
        if matched_genes:
            valid_proteins = set()
            for p in proteins:
                taxid = p.split('.')[0]
                if taxid in target_taxids:
                    valid_proteins.add(p)
                    
            for g in matched_genes:
                gene_to_target_proteins[g].update(valid_proteins)

out_map = {g: list(ps) for g, ps in gene_to_target_proteins.items() if ps}
print(f"Saved targets for {len(out_map)} genes.")
with open("pipeline/full_eggnog_pipeline/data/gene_to_target_proteins.json", "w") as f:
    json.dump(out_map, f)
