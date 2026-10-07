import sys
import subprocess
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from count_bits import FIELDS, history_to_bits, load_npz, save_npz, table_genes

def chunk_done(out_chunk):
    """COUNT ends a successful run with a '#SCORE' line."""
    if not os.path.exists(out_chunk) or os.path.getsize(out_chunk) == 0:
        return False
    with open(out_chunk, "rb") as f:
        f.seek(max(0, os.path.getsize(out_chunk) - 4096))
        tail = f.read().rstrip(b"\n").rsplit(b"\n", 1)[-1]
    return tail.startswith(b"#SCORE")

def run_chunk(chunk_id, tree_file, table_chunk, out_chunk, err_chunk, bits_chunk, genes):
    """Run COUNT on one chunk, then shrink its ~120 MB text output to packed
    bits (~1 MB) and delete the text, so peak disk is ~one text output per worker."""
    if not chunk_done(out_chunk):
        cmd = [
            # SerialGC + minimal JIT threads keep each JVM at ~1 core (default would spawn
            # GC threads proportional to the 128 host cores).
            "java", "-Xmx6g", "-XX:+UseSerialGC", "-XX:CICompilerCount=2",
            "-cp", "tools/CountXXV.jar", "count.model.Parsimony",
            "-gain", "1000", "-loss", "1",
            "-history", "true", "-ancestral", "false",
            tree_file, table_chunk
        ]
        with open(out_chunk, "w") as out_f, open(err_chunk, "w") as err_f:
            subprocess.run(cmd, stdout=out_f, stderr=err_f, check=True)
        if not chunk_done(out_chunk):
            raise RuntimeError(f"Count chunk {chunk_id} incomplete, see {err_chunk}")
    # COUNT numbers families from 0 within each chunk = row order of this chunk.
    n_nodes, bits = history_to_bits(out_chunk, len(genes))
    save_npz(bits_chunk, genes, n_nodes, bits)
    os.remove(out_chunk)
    if os.path.exists(err_chunk): os.remove(err_chunk)

def main():
    table_file = sys.argv[1]
    tree_file = sys.argv[2]
    out_file = sys.argv[3]  # .npz of packed presence/gain/loss bits (see count_bits.py)
    # Concurrency = cores reserved by SGE (NSLOTS); was hardcoded to 100.
    workers = int(os.environ.get("COUNT_WORKERS", os.environ.get("NSLOTS", 16)))
    n_chunks = int(os.environ.get("COUNT_CHUNKS", 100))  # many small chunks -> good load balance

    print(f"Splitting {table_file} into {n_chunks} chunks ({workers} concurrent)...")
    with open(table_file) as f:
        header = next(f)
        lines = f.readlines()
    genes = table_genes(table_file)

    chunk_size = max(1, -(-len(lines) // n_chunks))
    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]

    chunk_files = []
    for i, chunk_lines in enumerate(chunks):
        c_file = f"{table_file}.chunk{i}.tsv"
        o_file = f"{out_file}.chunk{i}.tsv"
        e_file = f"{out_file}.chunk{i}.err"
        b_file = f"{out_file}.chunk{i}.npz"
        chunk_genes = genes[i * chunk_size:(i + 1) * chunk_size]
        chunk_files.append((i, c_file, o_file, e_file, b_file, chunk_genes))

    # Resume: chunks already converted by an earlier (e.g. killed) run are reused.
    todo = [c for c in chunk_files if not os.path.exists(c[4])]
    todo_ids = {c[0] for c in todo}
    for (i, c_file, *_), chunk_lines in zip(chunk_files, chunks):
        if i in todo_ids:
            with open(c_file, "w") as f:
                f.write(header)
                f.writelines(chunk_lines)
    del lines, chunks

    print(f"Launching {len(todo)} parallel Count processes "
          f"({len(chunk_files) - len(todo)} chunks already done)...")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_chunk, i, tree_file, c_file, o_file, e_file, b_file, g)
                   for i, c_file, o_file, e_file, b_file, g in todo]
        for future in as_completed(futures):
            future.result()

    print("Merging results...")
    n_nodes, parts = None, {f: [] for f in FIELDS}
    for i, c_file, o_file, e_file, b_file, chunk_genes in chunk_files:
        g, n, bits = load_npz(b_file)
        if g != chunk_genes or (n_nodes is not None and n != n_nodes):
            sys.exit(f"Chunk {i} ({b_file}) does not match table.tsv / other chunks")
        n_nodes = n
        for f in FIELDS:
            parts[f].append(bits[f])
    save_npz(out_file, genes, n_nodes,
             {f: np.concatenate(parts[f]) for f in FIELDS})

    for i, c_file, o_file, e_file, b_file, _ in chunk_files:
        for p in (c_file, b_file):
            if os.path.exists(p): os.remove(p)

    print(f"Parallel Dollo Parsimony complete! -> {out_file} "
          f"({len(genes)} families x {n_nodes} nodes)")

if __name__ == "__main__":
    main()
