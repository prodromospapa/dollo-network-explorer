"""Human-referenced protein alignments for the 196-species TCS dataset.

For every gene in orthogroups.pkl:
  * the human reference is the UniProt (reviewed, canonical) sequence for the
    gene symbol; when the symbol is not in UniProt the longest human isoform
    in the orthogroup is used. Pfam domains are called on this exact sequence
  * every other species contributes ONE sequence: the orthogroup member with
    the best DIAMOND bitscore against the human reference
  * the rows are aligned with MAFFT and written, with the human Pfam
    domains, to <out>/<GENE>.json.gz for the explorer's alignment viewer

Stages (each resumable, outputs under --work):
  refs     gene -> orthogroup, human reference, needed protein ids
  prep     one input FASTA per gene (single pass over the proteomes)
  domains  Pfam-A hmmsearch on the human references (needs Pfam-A.hmm and
           Pfam-A.clans.tsv.gz from the Pfam FTP in --work)
  align    DIAMOND best hit per species + MAFFT, in parallel
  inject   merge domains.json into alignments built before it existed
"""
import argparse
import gzip
import json
import pickle
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

SCAPER_NEW = Path("/mnt/home2/prodromosp/scaper_new")
PROTEOMES = Path("/mnt/home2/prodromosp/orthofinder_scaper/TCS")
HUMAN = "EP00074_Homo_sapiens"


def species_of(col):
    """'EP00002_Diphylleia_rotans' -> 'Diphylleia_rotans' (tree leaf name)."""
    return "_".join(col.split("_")[1:])


def safe_name(gene):
    return gene.replace("/", "_")


# ---------------------------------------------------------------- refs
def stage_refs(work, uniprot_tsv):
    genes = list(pickle.load(open(SCAPER_NEW / "orthogroups.pkl", "rb")).index)
    prot2gene = pickle.load(open(SCAPER_NEW / "protein_dict.pkl", "rb"))

    # Same rule that built orthogroups.pkl: a gene belongs to the first
    # orthogroup (file order) that holds one of its human proteins.
    gene_og, og_members = {}, {}
    with open(SCAPER_NEW / "Orthogroups.tsv") as fh:
        cols = fh.readline().rstrip("\n").split("\t")[1:]
        hi = cols.index(HUMAN)
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            og, cells = parts[0], parts[1:] + [""] * (len(cols) + 1 - len(parts))
            human = [p.strip() for p in cells[hi].split(",") if p.strip()]
            if not human:
                continue
            claimed = False
            for p in human:
                g = prot2gene.get(p, p)
                if g not in gene_og:
                    gene_og[g] = og
                    claimed = True
            if claimed:
                og_members[og] = {species_of(c): [p.strip() for p in cell.split(",") if p.strip()]
                                  for c, cell in zip(cols, cells) if cell.strip()}

    # UniProt canonical by primary symbol, then by unambiguous synonym
    by_primary, by_syn = {}, defaultdict(set)
    with gzip.open(uniprot_tsv, "rt") as fh:
        fh.readline()
        for line in fh:
            acc, prim, syn, _len, seq = line.rstrip("\n").split("\t")
            if prim:
                for p in prim.split(";"):
                    by_primary.setdefault(p.strip(), (acc, seq))
            for s in syn.split():
                by_syn[s.strip(";")].add((acc, seq))

    refs, missing = {}, []
    for g in genes:
        og = gene_og.get(g)
        if og is None:
            missing.append(g)
            continue
        hit = by_primary.get(g)
        if hit is None and len(by_syn.get(g, ())) == 1:
            hit = next(iter(by_syn[g]))
        refs[g] = {"og": og, "acc": hit[0] if hit else None, "seq": hit[1] if hit else None,
                   "human_isoforms": og_members[og].get(species_of(HUMAN), [])}
    print(f"refs: {len(refs)} genes, {sum(1 for r in refs.values() if r['acc'])} with UniProt, "
          f"{len(missing)} without orthogroup: {missing[:10]}")
    with open(work / "refs.json", "w") as fh:
        json.dump(refs, fh)
    with open(work / "og_members.json", "w") as fh:
        json.dump({og: og_members[og] for og in {r["og"] for r in refs.values()}}, fh)


# ---------------------------------------------------------------- domains
def stage_domains(work, cpu):
    """Pfam-A domains on each gene's human reference (the first record of its
    input FASTA), with Pfam gathering thresholds and clan overlap resolution
    -- the same rules InterPro uses for its Pfam matches."""
    hmm = work / "Pfam-A.hmm"
    refs_fa = work / "human_refs.fa"
    with open(refs_fa, "w") as out:
        for fa in sorted((work / "in").glob("*.fa")):
            with open(fa) as fh:
                fh.readline()
                out.write(f">{fa.stem}\n{fh.readline()}")
    dom = work / "pfam.domtbl"
    subprocess.run(["hmmsearch", "--cut_ga", "--cpu", str(cpu), "--noali", "-o", "/dev/null",
                    "--domtblout", str(dom), str(hmm), str(refs_fa)], check=True)

    clan, desc = {}, {}
    with gzip.open(work / "Pfam-A.clans.tsv.gz", "rt") as fh:
        for line in fh:
            acc, cl, _cln, _id, d = line.rstrip("\n").split("\t")
            clan[acc] = cl or acc
            desc[acc] = d
    hits = defaultdict(list)
    for line in open(dom):
        if line.startswith("#"):
            continue
        f = line.split()
        gene, name, acc = f[0], f[3], f[4].split(".")[0]
        hits[gene].append({"acc": acc, "id": name, "name": desc.get(acc, name),
                           "start": int(f[19]), "end": int(f[20]), "e": float(f[12])})
    out = {}
    for gene, hs in hits.items():
        kept = []
        for h in sorted(hs, key=lambda h: h["e"]):
            # Drop a weaker hit overlapping a kept one from the same clan
            if any(clan.get(k["acc"]) == clan.get(h["acc"]) and h["start"] <= k["end"] and k["start"] <= h["end"]
                   for k in kept):
                continue
            kept.append(h)
        out[gene] = sorted(({k: v for k, v in h.items() if k != "e"} for h in kept), key=lambda h: h["start"])
    json.dump(out, open(work / "domains.json", "w"))
    print(f"domains: {sum(map(len, out.values()))} Pfam domains on {len(out)} genes")


def stage_inject(work, out):
    """Write domains.json into alignment files that were built without it."""
    doms = json.load(open(work / "domains.json"))
    n = 0
    for f in sorted(out.glob("*.json.gz")):
        with gzip.open(f, "rt") as fh:
            d = json.load(fh)
        new = doms.get(safe_name(d["g"]), [])
        if d.get("dom") == new:
            continue
        d["dom"] = new
        with gzip.open(f, "wt", compresslevel=9) as fh:
            json.dump(d, fh, separators=(",", ":"))
        n += 1
    print(f"inject: updated {n} files")


# ---------------------------------------------------------------- prep
def read_fasta(path):
    name, seq = None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name:
                    yield name, "".join(seq)
                name, seq = line[1:].split()[0], []
            else:
                seq.append(line.strip())
    if name:
        yield name, "".join(seq)


def stage_prep(work):
    refs = json.load(open(work / "refs.json"))
    members = json.load(open(work / "og_members.json"))
    needed = set()
    for og in {r["og"] for r in refs.values()}:
        for ids in members[og].values():
            needed.update(ids)
    print(f"prep: loading {len(needed):,} sequences from {PROTEOMES}")
    seqs = {}
    for fa in sorted(PROTEOMES.glob("*.fasta")):
        for name, s in read_fasta(fa):
            if name in needed:
                seqs[name] = s.replace("*", "")
    indir = work / "in"
    indir.mkdir(exist_ok=True)
    for g, r in refs.items():
        ref_seq = r["seq"] or max((seqs.get(p, "") for p in r["human_isoforms"]), key=len)
        ref_id = r["acc"] or max(r["human_isoforms"], key=lambda p: len(seqs.get(p, "")))
        with open(indir / f"{safe_name(g)}.fa", "w") as fh:
            fh.write(f">REF|Homo_sapiens|{ref_id}\n{ref_seq}\n")
            for sp, ids in members[r["og"]].items():
                if sp == species_of(HUMAN):
                    continue
                for p in ids:
                    if p in seqs:
                        fh.write(f">{sp}|{p}\n{seqs[p]}\n")
    print(f"prep: wrote {len(refs)} inputs")


# ---------------------------------------------------------------- align
def align_gene(args):
    gene, infile, outfile, dom, ref_info = args
    recs = list(read_fasta(infile))
    full = {}
    for line in open(infile):
        if line.startswith(">"):
            hdr = line[1:].strip()
            full[hdr.split()[0]] = hdr
    ref_name, ref_seq = recs[0]
    by_sp = defaultdict(list)
    for name, s in recs[1:]:
        sp, pid = name.split("|", 1)
        by_sp[sp].append((pid, s))

    chosen = {sp: c[0] for sp, c in by_sp.items() if len(c) == 1}
    multi = {sp: c for sp, c in by_sp.items() if len(c) > 1}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        if multi:
            q, d = td / "q.fa", td / "d.fa"
            q.write_text(f">ref\n{ref_seq}\n")
            with open(d, "w") as fh:
                for sp, c in multi.items():
                    for pid, s in c:
                        fh.write(f">{sp}|{pid}\n{s}\n")
            res = subprocess.run(
                ["diamond", "blastp", "-q", str(q), "-d", str(d), "--sensitive", "--threads", "1",
                 "--max-target-seqs", "0", "--evalue", "10", "--outfmt", "6", "sseqid", "bitscore", "--quiet"],
                capture_output=True, text=True)
            best = {}
            for line in res.stdout.splitlines():
                sid, bits = line.split("\t")
                sp, pid = sid.split("|", 1)
                if float(bits) > best.get(sp, ("", -1))[1]:
                    best[sp] = (pid, float(bits))
            seq_of = {(sp, pid): s for sp, c in multi.items() for pid, s in c}
            for sp in multi:
                if sp in best:
                    chosen[sp] = (best[sp][0], seq_of[(sp, best[sp][0])])
                else:
                    # No detectable hit: keep the longest member so the row is not lost
                    chosen[sp] = max(multi[sp], key=lambda c: len(c[1]))

        inp = td / "aln_in.fa"
        with open(inp, "w") as fh:
            fh.write(f">Homo_sapiens\n{ref_seq}\n")
            for sp, (pid, s) in chosen.items():
                fh.write(f">{sp}\n{s}\n")
        if not chosen:
            aligned = [("Homo_sapiens", ref_seq)]
        else:
            # --auto picks L-INS-i / FFT-NS-i, which takes minutes on long proteins;
            # past ~40k residues FFT-NS-2 is ~10x faster and close in quality
            total = len(ref_seq) + sum(len(s) for _, s in chosen.values())
            mode = ["--auto"] if total < 40000 else ["--retree", "2", "--maxiterate", "0"]
            res = subprocess.run(["mafft", *mode, "--thread", "1", "--quiet", "--anysymbol", str(inp)],
                                 capture_output=True, text=True)
            if res.returncode != 0:
                return gene, f"mafft failed: {res.stderr[:200]}"
            tmp = td / "aln.fa"
            tmp.write_text(res.stdout)
            aligned = [(n, s.upper()) for n, s in read_fasta(tmp)]

    rows = []
    for sp, s in aligned:
        pid = ref_name.split("|")[2] if sp == "Homo_sapiens" else chosen[sp][0]
        rows.append({"sp": sp, "id": pid, "s": s})
    out = {"g": gene, "og": ref_info["og"], "ref": {"acc": ref_info["acc"], "len": len(ref_seq)},
           "dom": dom, "rows": rows}
    with gzip.open(outfile, "wt", compresslevel=9) as fh:
        json.dump(out, fh, separators=(",", ":"))
    return gene, None


def stage_align(work, out, jobs, genes=None):
    refs = json.load(open(work / "refs.json"))
    doms = json.load(open(work / "domains.json")) if (work / "domains.json").exists() else {}
    out.mkdir(parents=True, exist_ok=True)
    todo = []
    for g, r in refs.items():
        if genes and g not in genes:
            continue
        of = out / f"{safe_name(g)}.json.gz"
        if of.exists() and not genes:
            continue
        todo.append((g, str(work / "in" / f"{safe_name(g)}.fa"), str(of), doms.get(safe_name(g), []), r))
    # Big inputs first so the slow tail overlaps with the many small ones
    todo.sort(key=lambda t: -Path(t[1]).stat().st_size)
    print(f"align: {len(todo)} genes with {jobs} workers")
    fails = []
    with ProcessPoolExecutor(jobs) as ex:
        for k, f in enumerate(as_completed([ex.submit(align_gene, t) for t in todo]), 1):
            g, err = f.result()
            if err:
                fails.append((g, err))
            if k % 200 == 0:
                print(f"  {k}/{len(todo)}  ({len(fails)} failed)", flush=True)
    if fails:
        json.dump(fails, open(work / "align_failures.json", "w"), indent=1)
    print(f"align: done, {len(fails)} failures")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["refs", "prep", "domains", "align", "inject"])
    ap.add_argument("--work", default="cache_alignments", type=Path)
    ap.add_argument("--out", default="alignments", type=Path)
    ap.add_argument("--uniprot", default="cache_alignments/uniprot_human.tsv.gz")
    ap.add_argument("--jobs", type=int, default=48)
    ap.add_argument("--cpu", type=int, default=32, help="hmmsearch threads")
    ap.add_argument("--genes", nargs="*", help="align only these genes (overwrites)")
    a = ap.parse_args()
    a.work.mkdir(exist_ok=True)
    if a.stage == "refs":
        stage_refs(a.work, a.uniprot)
    elif a.stage == "domains":
        stage_domains(a.work, a.cpu)
    elif a.stage == "prep":
        stage_prep(a.work)
    elif a.stage == "inject":
        stage_inject(a.work, a.out)
    else:
        stage_align(a.work, a.out, a.jobs, set(a.genes) if a.genes else None)


if __name__ == "__main__":
    sys.exit(main())
