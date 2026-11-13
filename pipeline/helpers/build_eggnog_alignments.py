"""Human-referenced protein alignments for the EggNOG (12.5k species) dataset.

For each requested gene:
  * candidates are the gene's EggNOG family proteins (fastas.db) that pass the
    gene's PANTHER HMM at E < 1e-3 -- the same rule that made the presence
    matrix, so the rows are exactly the species marked present on the tree
  * each species contributes ONE sequence, its best DIAMOND hit to the human
    reference (UniProt reviewed canonical for the gene symbol)
  * rows are aligned with MAFFT L-INS-i (FFT-NS-2 above --linsi-max rows)
  * Pfam-A domains are called on the human reference (gathering thresholds,
    clan overlaps resolved)
Output: <out>/<GENE>.json.gz in the same format as build_tcs_alignments.py.

  python3 pipeline/helpers/build_eggnog_alignments.py CNOT11 SCAPER TTC5
"""
import argparse
import gzip
import json
import sqlite3
import subprocess
import tempfile
import zlib
from collections import defaultdict
from pathlib import Path

from build_tcs_alignments import read_fasta

DB = "pipeline/full_eggnog_pipeline/data/fastas.db"
GENE_TO_PTHR = "pipeline/full_eggnog_pipeline/data/gene_to_pthr.json"
PANTHER_BIN = ("junk/panther_eggnog_pipeline/hmms/target/famlib/rel/PANTHER19.0_altVersion/"
               "hmmscoring/PANTHER19.0/globals/binHmm")


def family_fasta(gene, work):
    """All EggNOG family proteins for the gene (cached: the DB is slow over NFS)."""
    path = work / f"{gene}.all.fa"
    if not path.exists():
        conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        rows = conn.execute("SELECT s.prot_id, s.seq FROM gene_map g JOIN seqs s ON s.prot_id = g.prot_id "
                            "WHERE g.gene = ?", (gene,)).fetchall()
        conn.close()
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            for pid, z in rows:
                fh.write(f">{pid}\n{zlib.decompress(z).decode()}\n")
        tmp.rename(path)
    return dict(read_fasta(path))


def panther_pass(gene, pthr, fasta, work):
    hmm = work / f"{gene}.pthr.hmm"
    with open(hmm, "w") as fh:
        if subprocess.run(["hmmfetch", PANTHER_BIN, f"{pthr}.orig.30.pir"], stdout=fh,
                          stderr=subprocess.DEVNULL).returncode != 0:
            fh.seek(0)
            fh.truncate()
            subprocess.run(["hmmfetch", PANTHER_BIN, f"{pthr}.mag.pir"], stdout=fh, check=True)
    tbl = work / f"{gene}.pthr.tbl"
    subprocess.run(["hmmsearch", "--cpu", "8", "--noali", "-o", "/dev/null", "--domtblout", str(tbl),
                    str(hmm), str(fasta)], check=True)
    keep = set()
    for line in open(tbl):
        if not line.startswith("#"):
            f = line.split()
            if float(f[6]) < 1e-3:
                keep.add(f[0])
    return keep


def human_reference(gene, uniprot_tsv):
    syn = []
    with gzip.open(uniprot_tsv, "rt") as fh:
        fh.readline()
        for line in fh:
            acc, prim, syns, _len, seq = line.rstrip("\n").split("\t")
            if gene in [p.strip() for p in prim.split(";")]:
                return acc, seq
            if gene in syns.replace(";", " ").split():
                syn.append((acc, seq))
    return syn[0] if len(syn) == 1 else (None, None)


def pfam_domains(acc, seq, work):
    fa, tbl = work / f"ref_{acc}.fa", work / f"ref_{acc}.pfam.tbl"
    fa.write_text(f">ref\n{seq}\n")
    subprocess.run(["hmmscan", "--cut_ga", "--cpu", "8", "--noali", "-o", "/dev/null", "--domtblout", str(tbl),
                    "cache_alignments/Pfam-A.hmm", str(fa)], check=True)
    clan, desc = {}, {}
    with gzip.open("cache_alignments/Pfam-A.clans.tsv.gz", "rt") as fh:
        for line in fh:
            a, cl, _cln, _id, d = line.rstrip("\n").split("\t")
            clan[a], desc[a] = cl or a, d
    hits = []
    for line in open(tbl):
        if line.startswith("#"):
            continue
        f = line.split()
        a = f[1].split(".")[0]
        hits.append({"acc": a, "id": f[0], "name": desc.get(a, f[0]), "start": int(f[19]), "end": int(f[20]),
                     "e": float(f[12])})
    kept = []
    for h in sorted(hits, key=lambda h: h["e"]):
        if not any(clan[k["acc"]] == clan[h["acc"]] and h["start"] <= k["end"] and k["start"] <= h["end"] for k in kept):
            kept.append(h)
    return sorted(({k: v for k, v in h.items() if k != "e"} for h in kept), key=lambda h: h["start"])


def build(gene, work, out, uniprot_tsv, threads, linsi_max, max_iterate=1000):
    pthr = json.load(open(GENE_TO_PTHR)).get(gene)
    if not pthr:
        print(f"{gene}: no PANTHER family, skipped")
        return
    seqs = family_fasta(gene, work)
    keep = panther_pass(gene, pthr, work / f"{gene}.all.fa", work)
    acc, ref = human_reference(gene, uniprot_tsv)
    if not ref:
        print(f"{gene}: not in UniProt reviewed human, skipped")
        return
    by_sp = defaultdict(list)
    for pid in keep:
        sp = pid.split(".")[0]
        if sp != "9606":
            by_sp[sp].append(pid)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "q.fa").write_text(f">ref\n{ref}\n")
        with open(td / "d.fa", "w") as fh:
            for ids in by_sp.values():
                for pid in ids:
                    fh.write(f">{pid}\n{seqs[pid]}\n")
        res = subprocess.run(["diamond", "blastp", "-q", str(td / "q.fa"), "-d", str(td / "d.fa"), "--sensitive",
                              "--threads", str(threads), "--max-target-seqs", "0", "--evalue", "10",
                              "--outfmt", "6", "sseqid", "bitscore", "--quiet"],
                             capture_output=True, text=True, check=True)
        bits = {}
        for line in res.stdout.splitlines():
            sid, b = line.split("\t")
            bits[sid] = max(bits.get(sid, 0.0), float(b))
        # Best hit to human; a species with no DIAMOND hit keeps its longest HMM-passing protein
        chosen = {sp: max(ids, key=lambda p: (bits.get(p, -1.0), len(seqs[p]))) for sp, ids in by_sp.items()}

        with open(td / "in.fa", "w") as fh:
            fh.write(f">9606\n{ref}\n")
            for sp, pid in chosen.items():
                fh.write(f">{sp}\n{seqs[pid].replace('*', '')}\n")
        mode = (["--localpair", "--maxiterate", str(max_iterate)] if len(chosen) <= linsi_max
                else ["--retree", "2"])
        res = subprocess.run(["mafft", *mode, "--thread", str(threads), "--quiet", "--anysymbol", str(td / "in.fa")],
                             capture_output=True, text=True, check=True)
        (td / "aln.fa").write_text(res.stdout)
        aligned = list(read_fasta(td / "aln.fa"))

    rows = [{"sp": sp, "id": acc if sp == "9606" else chosen[sp], "s": s.upper()} for sp, s in aligned]
    rows.sort(key=lambda r: r["sp"] != "9606")
    data = {"g": gene, "og": f"EggNOG family / {pthr}", "ref": {"acc": acc, "len": len(ref)},
            "dom": pfam_domains(acc, ref, work), "rows": rows,
            "method": f"MAFFT {' '.join(mode)}; best DIAMOND hit to human per species among PANTHER-passing proteins"}
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f".{gene}.json.gz.tmp"
    with gzip.open(tmp, "wt", compresslevel=9) as fh:
        json.dump(data, fh, separators=(",", ":"))
    tmp.rename(out / f"{gene}.json.gz")   # atomic: the viewer never reads a half-written file
    print(f"{gene}: {len(rows)} rows x {len(rows[0]['s'])} columns, {len(data['dom'])} Pfam domains, "
          f"{sum(1 for sp in chosen if chosen[sp] not in bits)} species without a DIAMOND hit")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("genes", nargs="+")
    ap.add_argument("--work", default="cache_eggnog_aln", type=Path)
    ap.add_argument("--out", default="alignments_eggnog", type=Path)
    ap.add_argument("--uniprot", default="cache_alignments/uniprot_human.tsv.gz")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--linsi-max", type=int, default=1000, help="rows up to which L-INS-i is used")
    ap.add_argument("--maxiterate", type=int, default=1000,
                    help="L-INS-i refinement rounds (2 is much faster on large families)")
    a = ap.parse_args()
    a.work.mkdir(exist_ok=True)
    for g in a.genes:
        build(g, a.work, a.out, a.uniprot, a.threads, a.linsi_max, a.maxiterate)


if __name__ == "__main__":
    main()
