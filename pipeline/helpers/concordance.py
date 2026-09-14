#!/usr/bin/env python3
"""Helper for 03_concordance.sh.

Ranks every gene by its evolutionary co-loss with a seed gene, using the
Jaccard similarity of independently-lost tree branches:

    Jaccard = |seed losses ∩ candidate losses| / |seed losses ∪ candidate losses|

Two genes functionally co-evolve if they are independently lost on the same
branches of the tree. Jaccard is computed on LOSSES only (not gains),
because Dollo parsimony assigns each gene exactly one gain -- the
meaningful repeated, independent signal is loss. Dividing by the union
(not just one gene's loss count) keeps the score comparable between genes
with very different total numbers of losses.

Also flags whether each candidate is already a known ciliary gene per
SYSCILIA and CiliaCarta -- a face-validity check, not a statistic: if the
top-ranked candidates are disproportionately genes already known to be
ciliary, that's reassuring evidence the score is finding real signal. It
plays no role in the concordance score itself.

Usage:
    python3 concordance.py EVENTS_TSV SEED_GENE SYSCILIA_CSV CILIACARTA_CSV > results/SEED_concordance.tsv
"""
import csv
import sys
from collections import defaultdict


def load_loss_branches(events_path):
    """Return (all_genes, losses_by_gene): the full ordered list of genes
    seen in events.tsv, and {gene: set of nodeidx where that gene was
    lost}. Every gene is tracked explicitly -- including genes with ZERO
    losses (e.g. an almost-universally-present gene) -- so they still show
    up in the output with n_loss_candidate=0, matching the original awk
    version's behavior of initializing every gene's count up front rather
    than only creating an entry the first time a loss is seen."""
    all_genes = []
    seen = set()
    losses = defaultdict(set)
    with open(events_path) as f:
        next(f)  # header
        for line in f:
            gene, nodeidx, _presence, _gain, loss = line.rstrip("\n").split("\t")
            if gene not in seen:
                seen.add(gene)
                all_genes.append(gene)
            if loss == "1":
                losses[gene].add(nodeidx)
    return all_genes, losses


def read_gene_set(csv_path, column):
    """Set of gene symbols in a curated ciliary gene panel CSV."""
    genes = set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            symbol = (row.get(column) or "").strip()
            if symbol:
                genes.add(symbol)
    return genes


def score_candidates(all_genes, losses_by_gene, seed_gene):
    seed_losses = losses_by_gene.get(seed_gene, set())
    rows = []
    for gene in all_genes:
        if gene == seed_gene:
            continue
        candidate_losses = losses_by_gene.get(gene, set())
        shared = len(seed_losses & candidate_losses)
        union = len(seed_losses | candidate_losses)
        jaccard = shared / union if union > 0 else 0.0
        rows.append((gene, len(candidate_losses), shared, union, jaccard))
    # Sort by concordance descending; break ties alphabetically by gene name
    # so the output order is deterministic and reproducible run to run
    # (unlike awk's associative-array iteration order, which POSIX leaves
    # unspecified and so was never a meaningful ordering to begin with).
    rows.sort(key=lambda r: (-r[4], r[0]))
    return rows


def main():
    events_path, seed_gene, syscilia_csv, ciliacarta_csv = sys.argv[1:5]

    all_genes, losses_by_gene = load_loss_branches(events_path)
    if seed_gene not in all_genes:
        print(f"ERROR: Seed gene '{seed_gene}' not found in {events_path}", file=sys.stderr)
        sys.exit(1)

    syscilia_genes = read_gene_set(syscilia_csv, "resolved_symbol")
    ciliacarta_genes = read_gene_set(ciliacarta_csv, "Associated Gene Name")

    print(
        "candidate_gene\tn_loss_candidate\tshared_losses\tunion_losses\tconcordance"
        "\tin_syscilia\tin_ciliacarta"
    )
    for gene, n_loss, shared, union, jaccard in score_candidates(all_genes, losses_by_gene, seed_gene):
        in_syscilia = "yes" if gene in syscilia_genes else "no"
        in_ciliacarta = "yes" if gene in ciliacarta_genes else "no"
        # %.6g matches awk's default number formatting (6 significant
        # figures), so the numeric columns are byte-identical to the
        # original awk-based version.
        print(f"{gene}\t{n_loss}\t{shared}\t{union}\t{jaccard:.6g}\t{in_syscilia}\t{in_ciliacarta}")


if __name__ == "__main__":
    main()
