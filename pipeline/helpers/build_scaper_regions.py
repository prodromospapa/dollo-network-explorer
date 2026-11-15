"""Region-restricted alignments of the SCAPER orthogroup (TCS, 196 taxa) against
human SCAPER (UniProt Q9BY12), one sequence per species.

Input rows are the species/best-hit sequences of alignments_orthogroup/SCAPER.json.gz
(built by build_tcs_gene_alignments.py --dataset orthogroup SCAPER).

Regions (human residue numbering):
  N-domain        PF16501 (SCAPER_N) profile  -> hmmsearch envelope per protein, hmmalign to profile
  ZnF             PF12874 (zf-met) profile    -> same, as InterPro assigns the SCAPER zinc finger
  helical         413-769   (image: predicted alpha-helical domain)
                  -> column slice of the full alignment between human 413 and 769, then MAFFT L-INS-i
  alpha-solenoid  857-1395  (image: predicted alpha-solenoid CTD)
                  -> same slice + MAFFT L-INS-i procedure

Output: alignments_orthogroup/scaper_regions/<REGION>.fasta and summary.tsv

  python3 pipeline/helpers/build_scaper_regions.py
"""
import gzip
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PFAM = ROOT / "cache_alignments/Pfam-A.hmm"
SRC = ROOT / "alignments_orthogroup/SCAPER.json.gz"
OUT = ROOT / "alignments_orthogroup/scaper_regions"
HUMAN_SP = "Homo_sapiens"
PROFILES = {"N_domain_PF16501": "PF16501", "ZnF_PF12874": "PF12874"}
SLICES = {"helical_413-769": (413, 769), "alpha_solenoid_CTD_857-1395": (857, 1395)}


def load_rows():
    d = json.load(gzip.open(SRC, "rt"))
    rows = [{"sp": r["sp"], "id": r["id"], "s": r["s"].upper()} for r in d["rows"]]
    human = next(r for r in rows if r["sp"] == HUMAN_SP)
    return rows, human


def write_fasta(path, items):
    with open(path, "w") as fh:
        for name, seq in items:
            fh.write(f">{name}\n{seq}\n")


def read_fasta(path):
    name, buf = None, []
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith(">"):
            if name is not None:
                yield name, "".join(buf)
            name, buf = line[1:], []
        else:
            buf.append(line)
    if name is not None:
        yield name, "".join(buf)


def extract_hmm(acc, dest):
    """Copy one model (matched by ACC, e.g. PF16501) out of the Pfam-A.hmm library."""
    block, keep = [], False
    with open(PFAM) as fh, open(dest, "w") as out:
        for line in fh:
            block.append(line)
            if line.startswith("ACC") and line.split()[1].split(".")[0] == acc:
                keep = True
            if line.startswith("//"):
                if keep:
                    out.writelines(block)
                    return
                block = []
    raise SystemExit(f"{acc} not found in {PFAM}")


def profile_region(label, pfam_acc, rows, work):
    """Best SCAPER_N / zf-met envelope per species, aligned to the Pfam profile."""
    hmm = work / f"{pfam_acc}.hmm"
    extract_hmm(pfam_acc, hmm)
    fa = work / f"{label}.all.fa"
    write_fasta(fa, [(f"{r['sp']}|{r['id']}", r["s"].replace("-", "").replace("*", "")) for r in rows])
    tbl = work / f"{label}.domtbl"
    subprocess.run(["hmmsearch", "-E", "1e-3", "--domE", "1e-3", "--cpu", "8", "--noali", "-o", "/dev/null",
                    "--domtblout", str(tbl), str(hmm), str(fa)], check=True)
    best = {}
    for line in open(tbl):
        if line.startswith("#"):
            continue
        f = line.split()
        name, ievalue = f[0], float(f[12])
        env_from, env_to = int(f[19]), int(f[20])
        if name not in best or ievalue < best[name][0]:
            best[name] = (ievalue, env_from, env_to)
    seqs = dict(read_fasta(fa))
    env_fa = work / f"{label}.env.fa"
    items = [(n, seqs[n][best[n][1] - 1:best[n][2]]) for n in best]
    write_fasta(env_fa, items)
    res = subprocess.run(["hmmalign", "--outformat", "AFA", str(hmm), str(env_fa)],
                         capture_output=True, text=True, check=True)
    aligned = list(read_fasta_text(res.stdout))
    # Drop insert columns (lowercase in hmmalign AFA output): keep profile match states only
    if aligned:
        keep = [i for i in range(len(aligned[0][1])) if not any(s[i].islower() for _, s in aligned)]
        aligned = [(n, "".join(s[i] for i in keep).upper()) for n, s in aligned]
    info = {n: (best[n][1], best[n][2]) for n in best}
    return aligned, info, len(seqs) - len(best)


def read_fasta_text(text):
    name, buf = None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if name is not None:
                yield name, "".join(buf)
            name, buf = line[1:], []
        else:
            buf.append(line.strip())
    if name is not None:
        yield name, "".join(buf)


def human_columns(human_aligned):
    """Map human residue number (1-based) -> alignment column index."""
    cols, pos = {}, 0
    for i, c in enumerate(human_aligned):
        if c != "-":
            pos += 1
            cols[pos] = i
    return cols


def slice_region(rows, human, start, end, work, label):
    cols = human_columns(human["s"])
    c0, c1 = cols[start], cols[end]
    segs = []
    for r in rows:
        seg = r["s"][c0:c1 + 1].replace("-", "").replace(".", "")
        if seg:
            segs.append((f"{r['sp']}|{r['id']}", seg))
    hseg = human["s"][c0:c1 + 1].replace("-", "")
    segs = [(n, s) for n, s in segs if not n.startswith(f"{HUMAN_SP}|")]
    segs.insert(0, (f"{HUMAN_SP}|{human['id']}|{start}-{end}", hseg))
    fa = work / f"{label}.in.fa"
    write_fasta(fa, segs)
    res = subprocess.run(["mafft", "--localpair", "--maxiterate", "1000", "--thread", "8", "--quiet",
                          "--anysymbol", str(fa)], capture_output=True, text=True, check=True)
    aligned = list(read_fasta_text(res.stdout))
    keep = [i for i in range(len(aligned[0][1])) if any(s[i] != "-" for _, s in aligned)]
    aligned = [(n, "".join(s[i] for i in keep)) for n, s in aligned]
    return aligned, len(rows) - len(segs) + 1


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, human = load_rows()
    summary = []
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        for label, acc in PROFILES.items():
            aligned, info, missing = profile_region(label, acc, rows, work)
            write_fasta(OUT / f"SCAPER_{label}.fasta", aligned)
            hu = info.get(f"{HUMAN_SP}|{human['id']}")
            summary.append((label, acc, len(aligned), len(aligned[0][1]) if aligned else 0,
                            f"human {hu[0]}-{hu[1]}" if hu else "human not hit", missing))
            print(f"{label}: {len(aligned)} rows x {len(aligned[0][1])} cols, {missing} species without a hit")
        for label, (start, end) in SLICES.items():
            aligned, dropped = slice_region(rows, human, start, end, work, label)
            write_fasta(OUT / f"SCAPER_{label}.fasta", aligned)
            summary.append((label, "-", len(aligned), len(aligned[0][1]), f"human {start}-{end}", dropped))
            print(f"{label}: {len(aligned)} rows x {len(aligned[0][1])} cols, {dropped} species without residues in slice")
    with open(OUT / "summary.tsv", "w") as fh:
        fh.write("region\tpfam\trows\tcolumns\thuman_span\tspecies_without_region\n")
        for row in summary:
            fh.write("\t".join(map(str, row)) + "\n")


if __name__ == "__main__":
    main()
