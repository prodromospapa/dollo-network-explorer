import pickle
import gzip
import json
import csv
from pathlib import Path

DATA_DIR = Path("data")
OUT_DIR = Path("data/eggnog_dataset")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("1. Loading 19,758 human genes...")
with open("/home/prodromosp/scaper_new/orthogroups.pkl", "rb") as f:
    genes = list(pickle.load(f).index)

print("2. Mapping gene symbols to ENSP using mygene...")
ensp_cache = OUT_DIR / "gene_to_ensp.json"
gene_to_ensps = {}
if ensp_cache.exists():
    gene_to_ensps = json.loads(ensp_cache.read_text())
else:
    import mygene
    mg = mygene.MyGeneInfo()
    results = mg.querymany(genes, scopes='symbol', species='human', fields='ensembl.protein', as_dataframe=False)
    for r in results:
        query = r['query']
        ens = r.get('ensembl', {})
        prot = []
        if isinstance(ens, list):
            for e in ens:
                p = e.get('protein', [])
                if isinstance(p, str): p = [p]
                prot.extend(p)
        else:
            p = ens.get('protein', [])
            if isinstance(p, str): p = [p]
            prot.extend(p)
        if prot:
            gene_to_ensps[query] = prot
    ensp_cache.write_text(json.dumps(gene_to_ensps))

# Flatten mapping for fast lookup: "9606.ENSP..." -> gene symbol
# If multiple genes map to same ENSP, we map to a list of genes
protid_to_genes = {}
for gene, ensps in gene_to_ensps.items():
    for ensp in ensps:
        pid = f"9606.{ensp}"
        protid_to_genes.setdefault(pid, []).append(gene)

print(f"  Mapped {len(protid_to_genes)} unique EggNOG human protein IDs.")

print("3. Streaming e7.protein_families.tsv.gz...")
gene_presence = {g: set() for g in genes}
all_taxids = set()

with gzip.open("data/e7.protein_families.tsv.gz", "rt") as f:
    for i, line in enumerate(f):
        if i % 1000000 == 0:
            print(f"  Parsed {i} OGs...")
        parts = line.strip('\n').split('\t')
        if len(parts) < 6:
            continue
        proteins = parts[4].split(',')
        taxids = parts[5].split(',')
        
        # Check if any protein is in our human list
        matched_genes = set()
        for p in proteins:
            if p in protid_to_genes:
                for g in protid_to_genes[p]:
                    matched_genes.add(g)
        
        if matched_genes:
            tset = set(taxids)
            for g in matched_genes:
                # If a gene matches multiple OGs (shouldn't happen often per grep), take the union
                gene_presence[g].update(tset)
                all_taxids.update(tset)

print(f"  Found {len(all_taxids)} unique species (taxIDs) across all genes.")

print("4. Building presence matrix...")
sorted_taxids = sorted(list(all_taxids), key=lambda x: int(x))
import pandas as pd
import numpy as np

# Create numpy array for fast dataframe creation
matrix = np.zeros((len(genes), len(sorted_taxids)), dtype=np.int8)
taxid_to_idx = {t: i for i, t in enumerate(sorted_taxids)}

for r, g in enumerate(genes):
    for t in gene_presence[g]:
        matrix[r, taxid_to_idx[t]] = 1

df = pd.DataFrame(matrix, index=genes, columns=sorted_taxids)

# Save pkl
df.to_pickle(OUT_DIR / "eggnog_presence.pkl")
# Save csv
df.to_csv(OUT_DIR / "eggnog_presence.csv")
print(f"  Saved matrix to {OUT_DIR / 'eggnog_presence.pkl'} ({df.shape[0]} genes x {df.shape[1]} species)")

print("5. Generating taxonomic tree...")
from ete3 import NCBITaxa
ncbi = NCBITaxa()

# Filter taxids that ete3 knows
valid_taxids = []
for t in sorted_taxids:
    try:
        ncbi.get_taxid_translator([int(t)])
        valid_taxids.append(int(t))
    except Exception:
        pass

print(f"  {len(valid_taxids)}/{len(sorted_taxids)} taxids recognized by NCBITaxa.")
tree = ncbi.get_topology(valid_taxids)

# Resolve leaf names to strings (since cytoscape frontend expects string names)
# But wait, Dollo COUNT pipeline expects leaf names to match column names in the matrix (which are TaxID strings).
# So we must ensure leaf names are exactly the TaxID strings!
for leaf in tree.get_leaves():
    leaf.name = str(leaf.name)

tree.write(outfile=str(OUT_DIR / "eggnog_tree.newick"), format=1)
print(f"  Saved tree to {OUT_DIR / 'eggnog_tree.newick'}")

print("Done!")
