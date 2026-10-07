import numpy as np
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from count_bits import load_npz, unpack

bits_file = sys.argv[1]  # packed COUNT history bits from parallel_count.py
out_bin = sys.argv[2]
out_npy = sys.argv[3]
out_idx = sys.argv[4]
out_loss = sys.argv[5]

table_genes, n_nodes, bits = load_npz(bits_file)
# Genes in name order, as before (out_idx / Leiden / HTML index by this order).
order = sorted(range(len(table_genes)), key=table_genes.__getitem__)
genes = [table_genes[i] for i in order]

print("Building matrix...", file=sys.stderr)
M = unpack(bits["loss"], n_nodes)[order].astype(np.float32)

print("Computing Jaccard...", file=sys.stderr)
intersection = M @ M.T
row_sums = M.sum(axis=1)
union = row_sums[:, None] + row_sums[None, :] - intersection

with np.errstate(divide='ignore', invalid='ignore'):
    jaccard = intersection / union
    jaccard[np.isnan(jaccard)] = 0.0

print("Saving binary partners...", file=sys.stderr)
Path(out_bin).parent.mkdir(parents=True, exist_ok=True)
partner_dtype = np.dtype([("p", "<u4"), ("score", "<f4")])  # == struct "<If"
with open(out_bin, "wb") as fh:
    for i in range(len(genes)):
        row = jaccard[i]
        partners = np.flatnonzero(row > 0.1)
        partners = partners[partners != i]
        # Highest score first; ties keep ascending partner index.
        partners = partners[np.argsort(-row[partners], kind="stable")][:250]
        rec = np.empty(len(partners), dtype=partner_dtype)
        rec["p"], rec["score"] = partners, row[partners]
        fh.write(np.uint32(len(partners)).astype("<u4").tobytes())
        fh.write(rec.tobytes())

np.save(out_npy, jaccard)
np.save(out_loss, row_sums)

with open(out_idx, "w") as fh:
    fh.write("idx\tfid\tname\n")
    for i, g in enumerate(genes):
        fh.write(f"{i}\t{g}\t{g}\n")

print("Done computing Jaccard.", file=sys.stderr)
