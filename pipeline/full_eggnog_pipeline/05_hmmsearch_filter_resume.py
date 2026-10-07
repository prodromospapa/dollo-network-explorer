"""Resumable, resource-capped PANTHER HMM filter (step 3/8).

Skips every gene already present in ANY data/surviving_species*.tsv (the
original parallel run plus the chunk runs) and writes new results to
data/surviving_species_resume.tsv, which 06_build_filtered_matrix.py picks up
via its surviving_species*.tsv glob.

Filtering logic (hmmfetch .orig.30.pir -> fallback .mag.pir, full-sequence
E-value < 1e-3, taxid = prot_id prefix) is identical to
05_hmmsearch_filter_parallel.py so results are consistent.

Worker count defaults to $NSLOTS (set by SGE) so we never use more cores than
the scheduler has reserved for us.
"""
import argparse
import glob
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DATA = Path("pipeline/full_eggnog_pipeline/data")
ORIG_DB = DATA / "fastas.db"
LOCAL_DB = Path("/tmp/fastas_prodromosp.db")
BIN_HMM = Path("junk/panther_eggnog_pipeline/hmms/target/famlib/rel/"
               "PANTHER19.0_altVersion/hmmscoring/PANTHER19.0/globals/binHmm")
OUT_TSV = DATA / "surviving_species_resume.tsv"

ap = argparse.ArgumentParser()
ap.add_argument("--workers", type=int, default=int(os.environ.get("NSLOTS", 8)))
ap.add_argument("--limit", type=int, default=0, help="only process N genes (testing)")
args = ap.parse_args()

db_path = LOCAL_DB if LOCAL_DB.exists() and LOCAL_DB.stat().st_size == ORIG_DB.stat().st_size else ORIG_DB
print(f"Workers: {args.workers} | DB: {db_path}", flush=True)
if not Path(str(BIN_HMM) + ".h3i").exists():
    sys.exit(f"HMM index not found: {BIN_HMM}.h3i")

with open(DATA / "gene_to_pthr.json") as f:
    gene_to_pthr = json.load(f)

# --- genes already done (any previous run) ---------------------------------
processed = set()
for tsv in glob.glob(str(DATA / "surviving_species*.tsv")):
    with open(tsv) as f:
        for line in f:
            if line.startswith("gene\t") or not line.strip():
                continue
            processed.add(line.split("\t", 1)[0])
print(f"Already processed: {len(processed)} genes", flush=True)

# --- remaining genes, largest first for good load balancing ----------------
conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
sizes = dict(conn.execute("SELECT gene, COUNT(*) FROM gene_map GROUP BY gene").fetchall())
conn.close()
todo = sorted((g for g in sizes if g not in processed), key=lambda g: -sizes[g])
if args.limit:
    todo = todo[-args.limit:]  # smallest ones, for quick tests
total_seqs = sum(sizes[g] for g in todo)
print(f"Remaining: {len(todo)} genes / {total_seqs:,} sequences", flush=True)

_local = __import__("threading").local()


def get_conn():
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    return _local.conn


def process_gene(gene):
    if gene not in gene_to_pthr:
        return gene, ["NONE"]
    rows = get_conn().execute(
        "SELECT s.prot_id, s.seq FROM seqs s JOIN gene_map g ON s.prot_id = g.prot_id WHERE g.gene = ?",
        (gene,)).fetchall()
    if not rows:
        return gene, ["NONE"]
    fasta = "".join(f">{pid}\n{zlib.decompress(seq).decode()}\n" for pid, seq in rows).encode()
    del rows

    pthr = gene_to_pthr[gene]
    with tempfile.TemporaryDirectory(prefix="pp_hmm_") as td:
        hmm, tbl = f"{td}/q.hmm", f"{td}/q.tbl"
        with open(hmm, "w") as fh:
            res = subprocess.run(["hmmfetch", str(BIN_HMM), f"{pthr}.orig.30.pir"], stdout=fh, stderr=subprocess.DEVNULL)
        if res.returncode != 0:
            with open(hmm, "w") as fh:
                subprocess.run(["hmmfetch", str(BIN_HMM), f"{pthr}.mag.pir"], stdout=fh, stderr=subprocess.DEVNULL)
        subprocess.run(["hmmsearch", "--cpu", "1", "--noali", "--domtblout", tbl, hmm, "-"],
                       input=fasta, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        passed = set()
        if os.path.exists(tbl):
            with open(tbl) as fh:
                for line in fh:
                    if line.startswith("#"):
                        continue
                    p = line.split()
                    if len(p) > 12:
                        try:
                            if float(p[6]) < 1e-3:
                                passed.add(p[0].split(".")[0])
                        except ValueError:
                            pass
    return gene, sorted(passed) or ["NONE"]


new_file = not OUT_TSV.exists() or OUT_TSV.stat().st_size == 0
t0, done_seqs = time.time(), 0
with open(OUT_TSV, "a") as out, ThreadPoolExecutor(max_workers=args.workers) as ex:
    if new_file:
        out.write("gene\ttaxid\n")
    futs = {ex.submit(process_gene, g): g for g in todo}
    for i, fut in enumerate(as_completed(futs), 1):
        gene, taxids = fut.result()
        # one write per gene -> a gene is either fully recorded or absent
        out.write("".join(f"{gene}\t{t}\n" for t in taxids))
        out.flush()
        done_seqs += sizes[gene]
        if i % 50 == 0 or i == len(todo):
            el = time.time() - t0
            eta = el / max(done_seqs, 1) * (total_seqs - done_seqs)
            print(f"[{time.strftime('%F %T')}] {i}/{len(todo)} genes, "
                  f"{100*done_seqs/max(total_seqs,1):.1f}% seqs, elapsed {el/3600:.1f}h, "
                  f"ETA ~{eta/3600:.1f}h", flush=True)

print("Filtering complete!", flush=True)
