#!/usr/bin/env python3
"""Per human gene, per eggNOG 7 family: the species that hold
  (a) a pairwise ORTHOLOG of the gene's human protein (eggNOG's precomputed
      speciation events, sp_events/event_index -- what eggnog-mapper uses), and
  (b) a member of the gene's eukaryote-level ORTHOGROUP in that family
      (OG at level 2759; if the family has no 2759 split for this protein,
      its broadest OG, normally the root 131567).

eggNOG 7 builds families per domain, so a multi-domain protein (e.g. SCAPER:
zf-met + SCAPER_N) has one result per family. This script keeps every family
separately; 02_build_tables.py applies the gene-level rule.

Output: one JSON line per gene ->
  {"gene", "proteins": [...], "families": {fam: {"size": n_proteins_in_family,
   "ortho": [taxids], "og": [taxids], "og_name": str}},
   "ortho_all_families": [taxids where ONE protein is an ortholog in every family]}
Only taxids that are leaves of the eggNOG species tree are kept.

Usage (repo root):
  python3 pipeline/eggnog_orthology/01_extract.py OUT.jsonl [--genes G1,G2,...] [--workers N]
"""
import argparse
import gzip
import json
import os
import re
import sqlite3
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

DB = "/home/prodromosp/old/duplications_new/eggnog_db/eggnog.db"
FAMILIES = "data/e7.protein_families.tsv.gz"
GENE_TO_ENSP = "data/eggnog_dataset/gene_to_ensp.json"
SPECIES_TREE = "data/eggnog_dataset/eggnog_tree.newick"
CACHE = Path("cache_eggnog_v7")
TAXID_NPY = CACHE / "protein_taxid.npy"
HUMAN_JSON = CACHE / "human_protein_ids.json"
FAMSIZE_JSON = CACHE / "family_sizes.json"

ROOT_LEVEL, EUK_LEVEL = "131567", "2759"


def dec(blob):
    """Delta-varint BLOB -> sorted int64 array (vectorised eggnog codec)."""
    if not blob:
        return np.empty(0, np.int64)
    a = np.frombuffer(blob, np.uint8)
    term = a < 0x80
    gid = np.concatenate(([0], np.cumsum(term)[:-1]))
    starts = np.flatnonzero(np.concatenate(([True], term[:-1])))
    pos = np.arange(a.size) - starts[gid]
    deltas = np.bincount(gid, weights=(a & 0x7F).astype(np.float64) * (2.0 ** (7 * pos)))
    return np.cumsum(np.rint(deltas).astype(np.int64))


def connect():
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    c.execute("PRAGMA mmap_size=268435456")
    return c


def prepare(gene_filter):
    """One-off caches: protein id -> taxid array, human protein ids, family sizes."""
    CACHE.mkdir(exist_ok=True)
    c = connect()
    if not TAXID_NPY.exists():
        t = time.time()
        n = c.execute("select max(id) from protein_names").fetchone()[0] + 1
        arr = np.full(n, -1, np.int32)
        cur = c.execute("select id, taxid from protein_names")
        while True:
            rows = cur.fetchmany(1_000_000)
            if not rows:
                break
            r = np.array(rows, dtype=np.int64)
            arr[r[:, 0]] = r[:, 1]
        np.save(TAXID_NPY, arr)
        print(f"protein->taxid cache: {n} ids, {time.time()-t:.0f}s", flush=True)
    if not HUMAN_JSON.exists():
        hum = {n: i for i, n in c.execute("select id, name from protein_names where taxid=9606")}
        HUMAN_JSON.write_text(json.dumps(hum))
    hum = json.loads(HUMAN_JSON.read_text())

    g2e = json.loads(Path(GENE_TO_ENSP).read_text())
    genes = {}
    for g, ensps in g2e.items():
        if gene_filter and g not in gene_filter:
            continue
        ids = [hum["9606." + e] for e in ensps if "9606." + e in hum]
        genes[g] = ids  # may be empty -> gene absent from eggNOG

    # families touched by these proteins -> their sizes (n proteins)
    need = set()
    allp = [p for ids in genes.values() for p in ids]
    for k in range(0, len(allp), 900):
        ch = allp[k:k + 900]
        for (ogs,) in c.execute(f"select ogs from prots where id in ({','.join('?'*len(ch))})", ch):
            for og in (ogs or "").split(","):
                if og:
                    need.add(og.split("@")[0])
    sizes = json.loads(FAMSIZE_JSON.read_text()) if FAMSIZE_JSON.exists() else {}
    if need - set(sizes):
        t = time.time()
        with gzip.open(FAMILIES, "rt") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                p = line.split("\t", 2)
                if p[0] in need:
                    sizes[p[0]] = int(p[1])
        FAMSIZE_JSON.write_text(json.dumps(sizes))
        print(f"family sizes: {len(sizes)} families, {time.time()-t:.0f}s", flush=True)
    return genes


_W = {}


def _init(leaves):
    _W["c"] = connect()
    _W["tax"] = np.load(TAXID_NPY, mmap_mode="r")
    _W["leaves"] = np.array(sorted(leaves), dtype=np.int64)
    _W["og_cache"] = {}
    _W["sizes"] = json.loads(FAMSIZE_JSON.read_text())


def _taxa(prot_ids):
    if len(prot_ids) == 0:
        return []
    t = np.unique(np.asarray(_W["tax"][np.asarray(prot_ids)], dtype=np.int64))
    return np.intersect1d(t, _W["leaves"], assume_unique=True).tolist()


def _og_taxa(og_name):
    cache = _W["og_cache"]
    if og_name not in cache:
        row = _W["c"].execute("select seqs from ogs where name=?", (og_name,)).fetchone()
        cache[og_name] = _taxa(dec(row[0])) if row and row[0] else []
        if len(cache) > 2000:
            cache.pop(next(iter(cache)))
    return cache[og_name]


def _pick_og(ogs_for_family):
    """eukaryote orthogroup: the 2759 OG; else the root OG; else any (broadest)."""
    by_level = {og.split("@")[1].split("|")[0]: og for og in ogs_for_family}
    for lvl in (EUK_LEVEL, ROOT_LEVEL):
        if lvl in by_level:
            return by_level[lvl]
    return sorted(ogs_for_family)[0]


def work(item):
    gene, prot_ids = item
    c = _W["c"]
    fams = defaultdict(lambda: {"ortho_prots": [], "ogs": []})
    for pid in prot_ids:
        row = c.execute("select ogs from prots where id=?", (pid,)).fetchone()
        for og in ((row[0] if row else "") or "").split(","):
            if og:
                fams[og.split("@")[0]]["ogs"].append(og)
        r = c.execute("select events from event_index where protein_id=?", (pid,)).fetchone()
        ev = dec(r[0]) if r else np.empty(0)
        for k in range(0, len(ev), 900):
            ch = [int(x) for x in ev[k:k + 900]]
            for name, s1, s2 in c.execute(
                    f"select name, side1, side2 from sp_events where i in ({','.join('?'*len(ch))})", ch):
                a = dec(s1)
                other = dec(s2) if np.searchsorted(a, pid) < len(a) and a[np.searchsorted(a, pid)] == pid else a
                fams[name]["ortho_prots"].append(other)

    out_f, per_fam_prots = {}, {}
    for fam, d in fams.items():
        prots = np.unique(np.concatenate(d["ortho_prots"])) if d["ortho_prots"] else np.empty(0, np.int64)
        prots = prots[~np.isin(prots, prot_ids)]
        per_fam_prots[fam] = prots
        og = _pick_og(sorted(set(d["ogs"]))) if d["ogs"] else None
        out_f[fam] = {"size": _W["sizes"].get(fam), "ortho": _taxa(prots),
                      "og": _og_taxa(og) if og else [], "og_name": og}
    # strict whole-protein proxy: the same protein is an ortholog in every family
    with_events = [p for p in per_fam_prots.values()]
    if with_events and all(len(p) for p in with_events):
        common = with_events[0]
        for p in with_events[1:]:
            common = np.intersect1d(common, p, assume_unique=True)
        all_fam = _taxa(common)
    else:
        all_fam = []
    return json.dumps({"gene": gene, "proteins": prot_ids, "families": out_f,
                       "ortho_all_families": all_fam})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--genes", default=None, help="comma list, or @file with one gene per line")
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    gf = None
    if a.genes:
        gf = set(Path(a.genes[1:]).read_text().split()) if a.genes.startswith("@") else set(a.genes.split(","))
    genes = prepare(gf)
    leaves = {int(x) for x in re.findall(r"[(,]([0-9]+)[:),]", Path(SPECIES_TREE).read_text())}

    done = set()
    if os.path.exists(a.out):  # resume
        with open(a.out) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["gene"])
                except Exception:
                    pass
    todo = [(g, p) for g, p in genes.items() if g not in done and p]
    missing = [g for g, p in genes.items() if not p]
    print(f"{len(genes)} genes, {len(missing)} without an eggNOG human protein, "
          f"{len(done)} done, {len(todo)} to do, {len(leaves)} tree leaves", flush=True)
    # big families first for load balance is unknown; shuffle deterministically instead
    todo.sort(key=lambda x: hash(x[0]) % 1000)
    t = time.time()
    with open(a.out, "a") as fo, Pool(a.workers, initializer=_init, initargs=(leaves,)) as pool:
        for i, line in enumerate(pool.imap_unordered(work, todo, chunksize=4), 1):
            fo.write(line + "\n")
            if i % 200 == 0 or i == len(todo):
                fo.flush()
                el = time.time() - t
                print(f"{i}/{len(todo)} genes  {el/60:.1f} min  eta {el/i*(len(todo)-i)/60:.1f} min", flush=True)
    Path(a.out + ".missing_genes.txt").write_text("\n".join(sorted(missing)) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
