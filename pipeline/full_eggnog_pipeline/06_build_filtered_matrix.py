import json
import glob
from pathlib import Path
from ete3 import Tree
from collections import defaultdict

print("Reading surviving species from all chunks...")
gene_to_taxids = defaultdict(set)

# Read the original file (if it exists) and the chunk files
tsv_files = glob.glob("pipeline/full_eggnog_pipeline/data/surviving_species*.tsv")
for tsv_file in tsv_files:
    print(f"Parsing {tsv_file}...")
    with open(tsv_file) as f:
        for line in f:
            if line.startswith("gene\t") or not line.strip():
                continue
            gene, taxid = line.strip().split("\t")
            if taxid != "NONE":
                gene_to_taxids[gene].add(taxid)

print("Extracting full Newick tree...")
tree = Tree("data/eggnog_dataset/eggnog_tree.newick", format=1)
species_list = [leaf.name for leaf in tree.get_leaves()]

out_dir = Path("cache_eggnog_full")
out_dir.mkdir(exist_ok=True)
tree.write(format=1, outfile=str(out_dir / "tree.newick"))

print(f"Matrix: {len(gene_to_taxids)} genes x {len(species_list)} species")
with open(out_dir / "table.tsv", "w") as f:
    header = ["Family"] + species_list
    f.write("\t".join(header) + "\n")
    for gene, taxids in gene_to_taxids.items():
        row = [gene]
        for sp in species_list:
            row.append("1" if sp in taxids else "0")
        f.write("\t".join(row) + "\n")

print("Saved table.tsv and tree.newick!")
