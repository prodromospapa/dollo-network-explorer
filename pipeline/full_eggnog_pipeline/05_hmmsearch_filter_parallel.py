import sys
import json
import subprocess
import sqlite3
import zlib
import shutil
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

print("Loading gene mapping...")
with open("pipeline/full_eggnog_pipeline/data/gene_to_pthr.json") as f:
    gene_to_pthr = json.load(f)

orig_db = "pipeline/full_eggnog_pipeline/data/fastas.db"
db_path = "/tmp/fastas_prodromosp.db"

if not os.path.exists(db_path):
    print(f"Copying {orig_db} to local NVMe SSD ({db_path}) for turbo speed... (This will take a minute or two)")
    shutil.copy2(orig_db, db_path)
    print("Copy complete!")

hmms_dir = Path("/home/prodromosp/dollo-network-explorer/junk/panther_eggnog_pipeline/hmms")
binHmm = list(hmms_dir.rglob("binHmm.h3m"))[0].with_suffix("")
out_tsv = "pipeline/full_eggnog_pipeline/data/surviving_species.tsv"

print("Fetching unique genes from DB...")
conn = sqlite3.connect(db_path)
genes = [r[0] for r in conn.execute("SELECT DISTINCT gene FROM gene_map").fetchall()]
conn.close()
print(f"Found {len(genes)} unique genes to filter.")

# Only process genes that aren't already in out_tsv
processed = set()
if os.path.exists(out_tsv):
    with open(out_tsv) as f:
        for line in f:
            if line.startswith("gene"): continue
            parts = line.strip().split("\t")
            if parts: processed.add(parts[0])
            
genes = [g for g in genes if g not in processed]
print(f"Remaining genes to filter: {len(genes)}")

def process_gene(gene):
    if gene not in gene_to_pthr:
        return [(gene, "NONE")]
        
    conn = sqlite3.connect(db_path)
    query = """
        SELECT s.prot_id, s.seq 
        FROM seqs s
        JOIN gene_map g ON s.prot_id = g.prot_id
        WHERE g.gene = ?
    """
    rows = conn.execute(query, (gene,)).fetchall()
    conn.close()
    
    if not rows: return [(gene, "NONE")]
    
    fasta_str = ""
    for prot_id, compressed_seq in rows:
        seq = zlib.decompress(compressed_seq).decode('utf-8')
        fasta_str += f">{prot_id}\n{seq}\n"
        
    pthr = gene_to_pthr[gene]
    hmm_name = f"{pthr}.orig.30.pir"
    temp_hmm = f"/tmp/hmm_{gene}.hmm"
    
    with open(temp_hmm, "w") as f:
        res = subprocess.run(["hmmfetch", str(binHmm), hmm_name], stdout=f, stderr=subprocess.DEVNULL)
        
    if res.returncode != 0:
        with open(temp_hmm, "w") as f:
            subprocess.run(["hmmfetch", str(binHmm), f"{pthr}.mag.pir"], stdout=f, stderr=subprocess.DEVNULL)
            
    tbl_path = f"/tmp/tbl_{gene}.txt"
    cmd_search = ["hmmsearch", "--cpu", "1", "--noali", "--domtblout", tbl_path, temp_hmm, "-"]
    
    subprocess.run(cmd_search, input=fasta_str.encode('utf-8'), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    passed_taxids = set()
    if Path(tbl_path).exists():
        with open(tbl_path) as f:
            for line in f:
                if line.startswith("#"): continue
                parts = line.split()
                if len(parts) > 12:
                    try:
                        if float(parts[6]) < 1e-3:
                            prot_id = parts[0]
                            taxid = prot_id.split('.')[0]
                            passed_taxids.add(taxid)
                    except ValueError:
                        pass
        Path(tbl_path).unlink()
    if Path(temp_hmm).exists(): Path(temp_hmm).unlink()
    
    if not passed_taxids:
        return [(gene, "NONE")]
    return [(gene, t) for t in passed_taxids]

print("Starting parallel HMM search...")
mode = "a" if os.path.exists(out_tsv) else "w"
with open(out_tsv, mode) as f_out:
    if mode == "w": f_out.write("gene\ttaxid\n")
    import multiprocessing
    workers = 40
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_gene, g): g for g in genes}
        for i, future in enumerate(as_completed(futures)):
            results = future.result()
            for r in results:
                f_out.write(f"{r[0]}\t{r[1]}\n")
            f_out.flush()
            if i % 100 == 0:
                print(f"Processed {i}/{len(genes)} genes...", flush=True)

print("Filtering complete!")
