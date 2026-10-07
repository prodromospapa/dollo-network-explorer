import gzip
import json
import sqlite3
import zlib
from collections import defaultdict
from pathlib import Path

print("Loading targets...")
with open("pipeline/full_eggnog_pipeline/data/gene_to_target_proteins.json") as f:
    gene_to_target_proteins = json.load(f)

print("Building unique target set...")
unique_targets = set()
gene_map_batch = []
for g, prots in gene_to_target_proteins.items():
    for p in prots:
        unique_targets.add(p)
        gene_map_batch.append((g, p))

db_path = "pipeline/full_eggnog_pipeline/data/fastas.db"
if Path(db_path).exists():
    Path(db_path).unlink()
    
conn = sqlite3.connect(db_path)
c = conn.cursor()
c.execute("CREATE TABLE seqs (prot_id TEXT PRIMARY KEY, seq BLOB)")
c.execute("CREATE TABLE gene_map (gene TEXT, prot_id TEXT)")

print(f"Inserting {len(gene_map_batch)} mapping rows...")
c.executemany("INSERT INTO gene_map VALUES (?, ?)", gene_map_batch)
conn.commit()

print("Creating index on gene_map...")
c.execute("CREATE INDEX idx_gene ON gene_map(gene)")
conn.commit()
gene_map_batch = None # free memory

batch = []
BATCH_SIZE = 100000
fasta_path = "data/e7.proteins.fa.gz"

print(f"Streaming FASTA to SQLite (Targeting {len(unique_targets)} unique proteins)...")
count = 0
found = 0

with gzip.open(fasta_path, "rt") as f:
    current_prot = None
    current_seq = []
    
    for line in f:
        if line.startswith(">"):
            if current_prot and current_prot in unique_targets:
                seq = "".join(current_seq)
                compressed = zlib.compress(seq.encode('utf-8'))
                batch.append((current_prot, compressed))
                found += 1
                if len(batch) >= BATCH_SIZE:
                    c.executemany("INSERT INTO seqs VALUES (?, ?)", batch)
                    conn.commit()
                    batch = []
                    print(f"Inserted {found} unique sequences...", flush=True)
                        
            current_prot = line[1:].strip().split()[0]
            current_seq = []
            count += 1
        else:
            if current_prot in unique_targets:
                current_seq.append(line.strip())
                
    if current_prot and current_prot in unique_targets:
        seq = "".join(current_seq)
        compressed = zlib.compress(seq.encode('utf-8'))
        batch.append((current_prot, compressed))

if batch:
    c.executemany("INSERT INTO seqs VALUES (?, ?)", batch)
    conn.commit()

conn.close()
print("Done extracting to fastas.db!")
