import sys
import json
import struct
import numpy as np
import pandas as pd

table_path = sys.argv[1]
layout_path = sys.argv[2]
gene_order_path = sys.argv[3]
out_path = sys.argv[4]

table = pd.read_csv(table_path, sep="\t", index_col=0)
genes = table.index.tolist()

with open(layout_path) as f:
    layout = json.load(f)

# If it's a full JSON from UI, get names. If it's just tsv, read it.
if gene_order_path.endswith(".json"):
    with open(gene_order_path) as f:
        data_js = json.load(f)
    gene_order = data_js["names"]
else:
    with open(gene_order_path) as f:
        # Assuming jaccard_genes.tsv
        f.readline() # header
        gene_order = [line.split('\t')[2].strip() for line in f]

leaves = [n for n in layout["nodes"] if n["is_leaf"]]
leaves = sorted(leaves, key=lambda n: n["y"])
species_order = [n["species"] for n in leaves]

import math
n_bytes_per_gene = math.ceil(len(species_order) / 8)

out = open(out_path, "wb")
out.write(b"DLTP")
out.write(struct.pack("<II", len(gene_order), len(species_order)))

for gene in gene_order:
    if gene in table.index:
        row = table.loc[gene]
        bits = [int(row.get(sp, 0)) for sp in species_order]
    else:
        bits = [0] * len(species_order)
        
    byte_arr = bytearray(n_bytes_per_gene)
    for i, b in enumerate(bits):
        if b:
            byte_arr[i // 8] |= (1 << (i % 8))
    out.write(byte_arr)

out.close()
