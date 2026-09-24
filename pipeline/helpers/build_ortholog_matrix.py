#!/usr/bin/env python3
"""Build the gene-level "pure ortholog" presence matrix, mirroring how
/home/prodromosp/scaper_new/co_evolution_table.py built the existing
gene-level orthogroup matrix (orthogroups_corHMM.csv / orthogroups.pkl) --
but from human_ortholog_presence_matrix.tsv (OrthoFinder's reconciled
speciation/duplication ortholog calls) instead of the raw orthogroup
membership matrix.

Reuses the exact same protein_id -> gene_symbol map
(protein_dict.pkl, keyed by e.g. "EP00074_Homo_sapiens_P000001") that the
orthogroup matrix was collapsed with, and the same "first isoform wins"
tie-break, so the two matrices end up with identical gene identity/order --
only the per-species presence calls differ. That's what makes the two
datasets a meaningful apples-to-apples toggle rather than two different
gene lists.

The gene set is then explicitly restricted to the existing orthogroup
matrix's exact gene list (--orthogroup-pkl, required), by user decision:
naively collapsing the ortholog TSV on its own produces a slightly
different, larger gene set (21,226 vs 19,758) for two reasons -- (1) it
legitimately includes ~1,255 human proteins with zero orthogroup membership
at all (trivial "present in human only" rows the old orthogroup collapse
dropped), and (2) it doesn't reproduce a pre-existing ~270-gene
orthogroup-processing-order collision quirk in the old orthogroups.pkl
build (fallback-description gene "names" colliding across unrelated
orthogroups). Rather than reconcile those, we just intersect down to the
orthogroup matrix's gene list so both datasets line up gene-for-gene.

Outputs (species rows x gene columns, "Species" header -- same shape as
orthogroups_corHMM.csv):
    orthologs_corHMM.csv
    orthologs.pkl           (gene rows x species columns, like orthogroups.pkl)

Usage:
    python3 build_ortholog_matrix.py \
        --ortholog-tsv /home/prodromosp/scaper_tree/230228_TCS_196taxa/human_ortholog_presence_matrix.tsv \
        --protein-dict /home/prodromosp/scaper_new/protein_dict.pkl \
        --orthogroup-pkl /home/prodromosp/scaper_new/orthogroups.pkl \
        --out-csv orthologs_corHMM.csv \
        --out-pkl orthologs.pkl
"""
import argparse
import csv
import pickle
import sys


def strip_ep_prefix(col):
    """'EP00002_Diphylleia_rotans' -> 'Diphylleia_rotans' (matches
    co_evolution_table.py's "_".join(col.split("_")[1:]) convention)."""
    return "_".join(col.split("_")[1:])


def load_protein_dict(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_description_to_protein_id(human_fasta_path):
    """Map raw protein description (as it appears after the RefSeq accession
    in the human FASTA header) -> first protein id with that description.
    Needed to resolve orthogroups.pkl gene-index entries that are
    themselves a raw description string rather than a resolved gene symbol
    (see resolve_keep_genes)."""
    desc_to_pid = {}
    with open(human_fasta_path) as f:
        for line in f:
            if not line.startswith(">"):
                continue
            parts = line[1:].strip().replace(" [Homo sapiens]", "").split(" ")
            protein_id, desc = parts[0], " ".join(parts[2:]).strip()
            desc_to_pid.setdefault(desc, protein_id)
    return desc_to_pid


def resolve_keep_genes(keep_genes, protein_to_gene, desc_to_pid):
    """orthogroups.pkl's index is mostly clean gene symbols, but ~46 entries
    are stale raw-description strings left over from an earlier, since-
    corrected protein_id -> gene_symbol resolution (see module docstring).
    For each such key, resolve it to the protein id whose FASTA description
    matches the key literally, then take THAT protein's current gene
    symbol as the row to copy presence values from. If that symbol is a
    different key already in keep_genes (a true duplicate -- the common
    case), this key's row is just a copy of that other row, and is only
    kept to preserve identity for genes with no other row under any name.

    Returns {orthogroup_key: gene_symbol_to_pull_ortholog_presence_from}."""
    resolved = set(protein_to_gene.values())
    resolved_ci = {g.upper(): g for g in resolved}  # case-insensitive fallback (e.g. GRIFIN/grifin)
    alias = {}
    for key in keep_genes:
        if key in resolved:
            alias[key] = key
            continue
        if key.upper() in resolved_ci:
            alias[key] = resolved_ci[key.upper()]
            continue
        pid = desc_to_pid.get(key)
        real_gene = protein_to_gene.get(pid) if pid else None
        alias[key] = real_gene if real_gene else key
    return alias


def build_gene_rows(ortholog_tsv_path, protein_to_gene, keep_genes, resolved_gene_to_keys):
    """Read human_ortholog_presence_matrix.tsv and collapse to one row per
    orthogroup-matrix gene key, first protein id seen wins (same tie-break
    as co_evolution_table.py's gene_to_row.setdefault).

    keep_genes is the existing orthogroup matrix's exact gene-key set to
    reproduce. resolved_gene_to_keys maps a resolved gene symbol -> the
    list of orthogroup-matrix keys that should receive a copy of its
    presence row (usually one key; more than one when several stale
    description-string keys and/or the real symbol key all ultimately
    resolve to the same gene -- see resolve_keep_genes)."""
    gene_to_row = {}
    unmapped = []
    skipped_not_in_keep = 0
    with open(ortholog_tsv_path, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        species_cols = header[3:]  # human_protein, refseq_accession, description, then species
        species = [strip_ep_prefix(c) for c in species_cols]
        for row in reader:
            protein_id = row[0]
            gene_name = protein_to_gene.get(protein_id)
            if gene_name is None:
                unmapped.append(protein_id)
                continue
            keys = resolved_gene_to_keys.get(gene_name)
            if not keys:
                skipped_not_in_keep += 1
                continue
            new_keys = [k for k in keys if k not in gene_to_row]
            if not new_keys:
                continue  # first isoform wins, already filled for every target key
            values = [int(v) for v in row[3:]]
            for k in new_keys:
                gene_to_row[k] = values
    print(f"  -> {skipped_not_in_keep} protein rows skipped (gene not in orthogroup matrix's gene set).")
    missing = keep_genes - gene_to_row.keys()
    if missing:
        print(f"  WARNING: {len(missing)} orthogroup-matrix genes had no row in the ortholog TSV at all: {sorted(missing)[:10]}...")
    return species, gene_to_row, unmapped


def write_csv(out_csv_path, species, gene_to_row):
    genes = list(gene_to_row.keys())
    with open(out_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Species"] + genes)
        for i, sp in enumerate(species):
            writer.writerow([sp] + [gene_to_row[g][i] for g in genes])
    return genes


def write_pkl(out_pkl_path, species, genes, gene_to_row):
    import pandas as pd

    df = pd.DataFrame(
        {g: gene_to_row[g] for g in genes}, index=species
    ).T
    df.columns = species
    with open(out_pkl_path, "wb") as f:
        pickle.dump(df, f)
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ortholog-tsv", required=True)
    parser.add_argument("--protein-dict", required=True)
    parser.add_argument("--human-fasta", required=True,
                         help="EP00074_Homo_sapiens.fasta -- needed to resolve the handful of "
                              "orthogroups.pkl gene-index entries that are stale raw "
                              "description strings rather than resolved gene symbols")
    parser.add_argument("--orthogroup-pkl", required=True,
                         help="Existing orthogroups.pkl -- also used to restrict the new "
                              "matrix to the same gene set (see module docstring)")
    parser.add_argument("--out-csv", default="orthologs_corHMM.csv")
    parser.add_argument("--out-pkl", default="orthologs.pkl")
    args = parser.parse_args()

    print("1. Loading protein_id -> gene_symbol map...")
    protein_to_gene = load_protein_dict(args.protein_dict)
    print(f"  -> {len(protein_to_gene)} protein IDs mapped.")

    print("2. Loading existing orthogroup matrix's gene set (to match)...")
    with open(args.orthogroup_pkl, "rb") as f:
        orthogroups = pickle.load(f)
    keep_genes = set(orthogroups.index)
    print(f"  -> {len(keep_genes)} genes to match.")

    print("2b. Resolving stale description-string keys in the orthogroup gene set...")
    desc_to_pid = build_description_to_protein_id(args.human_fasta)
    key_alias = resolve_keep_genes(keep_genes, protein_to_gene, desc_to_pid)
    n_aliased = sum(1 for k, v in key_alias.items() if k != v)
    print(f"  -> {n_aliased} keys aliased to a resolved gene symbol.")
    resolved_gene_to_keys = {}
    for key, resolved_gene in key_alias.items():
        resolved_gene_to_keys.setdefault(resolved_gene, []).append(key)

    print("3. Reading ortholog presence matrix and collapsing to gene level...")
    species, gene_to_row, unmapped = build_gene_rows(
        args.ortholog_tsv, protein_to_gene, keep_genes=keep_genes,
        resolved_gene_to_keys=resolved_gene_to_keys,
    )
    print(f"  -> {len(gene_to_row)} genes x {len(species)} species.")
    if unmapped:
        print(f"  WARNING: {len(unmapped)} protein IDs had no gene symbol mapping (skipped).")

    print(f"4. Writing {args.out_csv}...")
    genes = write_csv(args.out_csv, species, gene_to_row)

    print(f"5. Writing {args.out_pkl}...")
    df = write_pkl(args.out_pkl, species, genes, gene_to_row)

    print("6. Sanity check against existing orthogroup matrix...")
    assert set(genes) == keep_genes, "gene set mismatch after restriction -- should be impossible"
    import numpy as np
    ortho_totals = orthogroups.loc[genes].sum(axis=1)
    ortholog_totals = df.loc[genes].sum(axis=1)
    gap = ortho_totals - ortholog_totals
    print(f"  Genes: {len(genes)} (exact match with orthogroup matrix, as intended)")
    print(f"  Mean species/gene  -- orthogroup: {ortho_totals.mean():.1f}  ortholog: {ortholog_totals.mean():.1f}")
    print(f"  Median gap: {np.median(gap):.0f}")
    print(f"  Genes with gap >= 50: {(gap >= 50).sum()} ({100*(gap >= 50).mean():.0f}%)")
    print(f"  Max gap: {gap.max()}")

    print("Done.")


if __name__ == "__main__":
    main()
