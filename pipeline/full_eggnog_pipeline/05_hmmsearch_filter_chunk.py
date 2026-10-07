import sys
import json
import subprocess
import sqlite3
import zlib
import shutil
import os
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

parser = argparse.ArgumentParser()
parser.add_argument("--chunk", type=int, required=True, help="Chunk index (1-based)")
parser.add_argument("--workers", type=int, default=12, help="Number of workers"); parser.add_argument("--total-chunks", type=int, required=True, help="Total number of chunks")
args = parser.parse_args()

print("Loading gene mapping...")
with open("pipeline/full_eggnog_pipeline/data/gene_to_pthr.json") as f:
    gene_to_pthr = json.load(f)

orig_db = "pipeline/full_eggnog_pipeline/data/fastas.db"
# Use a chunk-specific tmp DB so if multiple run on same machine they don't clash, but ideally they run on different nodes
db_path = f"/tmp/fastas_prodromosp_chunk{args.chunk}.db"

if not os.path.exists(db_path):
    print(f"Copying {orig_db} to local NVMe SSD ({db_path})...")
    shutil.copy2(orig_db, db_path)

hmms_dir = Path("/home/prodromosp/dollo-network-explorer/junk/panther_eggnog_pipeline/hmms")
binHmm = list(hmms_dir.rglob("binHmm.h3m"))[0].with_suffix("")
out_tsv = f"pipeline/full_eggnog_pipeline/data/surviving_species_chunk{args.chunk}.tsv"

print("Fetching unique genes from DB...")
conn = sqlite3.connect(db_path)
genes = [r[0] for r in conn.execute("SELECT DISTINCT gene FROM gene_map ORDER BY gene").fetchall()]
conn.close()

# Only keep genes for this chunk
chunk_genes = [g for i, g in enumerate(genes) if i % args.total_chunks == (args.chunk - 1)]

# Filter out already processed
processed = set()
if os.path.exists(out_tsv):
    with open(out_tsv) as f:
        for line in f:
            if line.startswith("gene"): continue
            parts = line.strip().split("\t")
            if parts: processed.add(parts[0])
            
chunk_genes = [g for g in chunk_genes if g not in processed]
print(f"Chunk {args.chunk}/{args.total_chunks} has {len(chunk_genes)} genes left to process.")

def process_gene(gene):
    if gene not in gene_to_pthr: return [(gene, "NONE")]
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT s.prot_id, s.seq FROM seqs s JOIN gene_map g ON s.prot_id = g.prot_id WHERE g.gene = ?", (gene,)).fetchall()
    conn.close()
    if not rows: return [(gene, "NONE")]
    
    fasta_str = ""
    for prot_id, compressed_seq in rows:
        fasta_str += f">{prot_id}\n{zlib.decompress(compressed_seq).decode('utf-8')}\n"
        
    pthr = gene_to_pthr[gene]
    temp_hmm = f"/tmp/hmm_{gene}_c{args.chunk}.hmm"
    tbl_path = f"/tmp/tbl_{gene}_c{args.chunk}.txt"
    
    with open(temp_hmm, "w") as f:
        res = subprocess.run(["hmmfetch", str(binHmm), f"{pthr}.orig.30.pir"], stdout=f, stderr=subprocess.DEVNULL)
    if res.returncode != 0:
        with open(temp_hmm, "w") as f:
            subprocess.run(["hmmfetch", str(binHmm), f"{pthr}.mag.pir"], stdout=f, stderr=subprocess.DEVNULL)
            
    subprocess.run(["hmmsearch", "--cpu", "1", "--noali", "--domtblout", tbl_path, temp_hmm, "-"], input=fasta_str.encode('utf-8'), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    passed_taxids = set()
    if Path(tbl_path).exists():
        with open(tbl_path) as f:
            for line in f:
                if line.startswith("#"): continue
                parts = line.split()
                if len(parts) > 12:
                    try:
                        if float(parts[6]) < 1e-3: passed_taxids.add(parts[0].split('.')[0])
                    except ValueError: pass
        Path(tbl_path).unlink()
    if Path(temp_hmm).exists(): Path(temp_hmm).unlink()
    
    if not passed_taxids: return [(gene, "NONE")]
    return [(gene, t) for t in passed_taxids]

mode = "a" if os.path.exists(out_tsv) else "w"
with open(out_tsv, mode) as f_out:
    if mode == "w": f_out.write("gene\ttaxid\n")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_gene, g): g for g in chunk_genes}
        for i, future in enumerate(as_completed(futures)):
            for r in future.result(): f_out.write(f"{r[0]}\t{r[1]}\n")
            f_out.flush()
            if i % 100 == 0: print(f"Chunk {args.chunk}: Processed {i}/{len(chunk_genes)} genes...", flush=True)

print(f"Chunk {args.chunk} complete!")
