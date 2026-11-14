"""Human-referenced alignments for chosen genes of the 196-species TCS dataset,
for either presence definition the explorer offers:

  orthogroup  candidates per species = members of the gene's OrthoFinder
              orthogroup (same gene -> orthogroup rule as orthogroups.pkl,
              so rows = species marked present in data.json)
  ortholog    candidates per species = OrthoFinder's pairwise orthologs of
              the gene's human protein (Orthologues_EP00074_Homo_sapiens/),
              using the first isoform of the gene in
              human_ortholog_presence_matrix.tsv -- the same isoform
              build_ortholog_matrix.py took, so rows = species marked present
              in data.ortholog.json

Each species contributes ONE sequence, its best DIAMOND hit to the human
reference (UniProt reviewed canonical). Rows are aligned with MAFFT L-INS-i
and Pfam-A domains are called on the reference, as in
build_eggnog_alignments.py. Output format is the viewer's usual
<out>/<GENE>.json.gz.

  python3 pipeline/helpers/build_tcs_gene_alignments.py --dataset orthogroup TTC5 CNOT11 SCAPER
  python3 pipeline/helpers/build_tcs_gene_alignments.py --dataset ortholog   TTC5 CNOT11 SCAPER
"""
import argparse
import gzip
import json
import pickle
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from build_eggnog_alignments import human_reference, pfam_domains
from build_tcs_alignments import HUMAN, PROTEOMES, SCAPER_NEW, read_fasta, species_of

TCS = Path("/home/prodromosp/scaper_tree/230228_TCS_196taxa")
ORTHO_DIR = TCS / "Orthofinder_TCS_result/Orthologues/Orthologues_EP00074_Homo_sapiens"
ORTHO_TSV = TCS / "human_ortholog_presence_matrix.tsv"
HUMAN_SP = species_of(HUMAN)


def split_cell(cell):
    return [p.strip() for p in cell.split(",") if p.strip()]


def orthogroup_candidates(genes, prot2gene):
    """gene -> (orthogroup id, {species: [protein ids]}); first orthogroup in
    file order holding one of the gene's human proteins (build_tcs_alignments rule)."""
    out = {}
    with open(SCAPER_NEW / "Orthogroups.tsv") as fh:
        cols = fh.readline().rstrip("\n").split("\t")[1:]
        hi = cols.index(HUMAN)
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            og, cells = parts[0], parts[1:] + [""] * (len(cols) + 1 - len(parts))
            for p in split_cell(cells[hi]):
                g = prot2gene.get(p, p)
                if g in genes and g not in out:
                    out[g] = (og, {species_of(c): split_cell(cell) for c, cell in zip(cols, cells)
                                   if cell.strip() and species_of(c) != HUMAN_SP})
            if len(out) == len(genes):
                break
    return out


def ortholog_candidates(genes, prot2gene):
    """gene -> (human protein id, {species: [ortholog protein ids]})."""
    first = {}
    with open(ORTHO_TSV) as fh:
        fh.readline()
        for line in fh:
            pid = line.split("\t", 1)[0]
            g = prot2gene.get(pid)
            if g in genes and g not in first:
                first[g] = pid
    pid_gene = {p: g for g, p in first.items()}
    found = {g: defaultdict(list) for g in first}
    for f in sorted(ORTHO_DIR.glob("*.tsv")):
        with open(f) as fh:
            cols = fh.readline().rstrip("\n").split("\t")
            sp = species_of(cols[2])
            for line in fh:
                _og, hum, other = line.rstrip("\n").split("\t")
                for p in split_cell(hum):
                    if p in pid_gene:
                        found[pid_gene[p]][sp].extend(split_cell(other))
    return {g: (first[g], dict(found[g])) for g in first}


def load_sequences(ids):
    seqs = {}
    by_sp = defaultdict(set)
    for p in ids:
        by_sp[p.rsplit("_", 1)[0]].add(p)   # 'EP00002_Diphylleia_rotans_P009625' -> proteome file stem
    for stem, want in by_sp.items():
        for name, s in read_fasta(PROTEOMES / f"{stem}.fasta"):
            if name in want:
                seqs[name] = s.replace("*", "")
    return seqs


def build(gene, label, cands, seqs, work, out, uniprot, threads, dataset):
    acc, ref = human_reference(gene, uniprot)
    if not ref:
        print(f"{gene}: not in UniProt reviewed human, skipped")
        return
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "q.fa").write_text(f">ref\n{ref}\n")
        with open(td / "d.fa", "w") as fh:
            for ids in cands.values():
                for p in ids:
                    fh.write(f">{p}\n{seqs[p]}\n")
        res = subprocess.run(["diamond", "blastp", "-q", str(td / "q.fa"), "-d", str(td / "d.fa"), "--sensitive",
                              "--threads", str(threads), "--max-target-seqs", "0", "--evalue", "10",
                              "--outfmt", "6", "sseqid", "bitscore", "--quiet"],
                             capture_output=True, text=True, check=True)
        bits = {}
        for line in res.stdout.splitlines():
            sid, b = line.split("\t")
            bits[sid] = max(bits.get(sid, 0.0), float(b))
        # Best hit to human; a species with no DIAMOND hit keeps its longest candidate
        chosen = {sp: max(ids, key=lambda p: (bits.get(p, -1.0), len(seqs[p]))) for sp, ids in cands.items()}
        with open(td / "in.fa", "w") as fh:
            fh.write(f">{HUMAN_SP}\n{ref}\n")
            for sp, p in chosen.items():
                fh.write(f">{sp}\n{seqs[p]}\n")
        mode = ["--localpair", "--maxiterate", "1000"]
        res = subprocess.run(["mafft", *mode, "--thread", str(threads), "--quiet", "--anysymbol", str(td / "in.fa")],
                             capture_output=True, text=True, check=True)
        (td / "aln.fa").write_text(res.stdout)
        aligned = list(read_fasta(td / "aln.fa"))

    rows = [{"sp": sp, "id": acc if sp == HUMAN_SP else chosen[sp], "s": s.upper()} for sp, s in aligned]
    rows.sort(key=lambda r: r["sp"] != HUMAN_SP)
    what = ("orthogroup members" if dataset == "orthogroup"
            else f"OrthoFinder orthologs of {label}")
    data = {"g": gene, "og": label if dataset == "orthogroup" else f"Orthologs of {label}",
            "ref": {"acc": acc, "len": len(ref)}, "dom": pfam_domains(acc, ref, work), "rows": rows,
            "method": f"MAFFT {' '.join(mode)}; best DIAMOND hit to human per species among {what}"}
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f".{gene}.json.gz.tmp"
    with gzip.open(tmp, "wt", compresslevel=9) as fh:
        json.dump(data, fh, separators=(",", ":"))
    tmp.rename(out / f"{gene}.json.gz")
    print(f"{dataset} {gene} ({label}): {len(rows)} rows x {len(rows[0]['s'])} columns, "
          f"{len(data['dom'])} Pfam domains, {sum(1 for p in chosen.values() if p not in bits)} species without a DIAMOND hit")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("genes", nargs="+")
    ap.add_argument("--dataset", choices=["orthogroup", "ortholog"], required=True)
    ap.add_argument("--out", type=Path, help="default alignments_<dataset>/")
    ap.add_argument("--work", default="cache_tcs_aln", type=Path)
    ap.add_argument("--uniprot", default="cache_alignments/uniprot_human.tsv.gz")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    out = a.out or Path(f"alignments_{a.dataset}")
    a.work.mkdir(exist_ok=True)
    prot2gene = pickle.load(open(SCAPER_NEW / "protein_dict.pkl", "rb"))
    genes = set(a.genes)
    cands = (orthogroup_candidates if a.dataset == "orthogroup" else ortholog_candidates)(genes, prot2gene)
    for g in genes - cands.keys():
        print(f"{g}: no human protein found in the {a.dataset} data, skipped")
    seqs = load_sequences({p for _, c in cands.values() for ids in c.values() for p in ids})
    for g in a.genes:
        if g in cands:
            label, c = cands[g]
            build(g, label, c, seqs, a.work, out, a.uniprot, a.threads, a.dataset)


if __name__ == "__main__":
    main()
