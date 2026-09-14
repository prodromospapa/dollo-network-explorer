#!/usr/bin/env python3
"""Helper for 01_prepare_inputs.sh.

Converts the project's input files into the exact format COUNT requires:
  1. Tree: NEXUS -> Newick, with every species name single-quoted (the
     Newick spec treats an unquoted underscore as whitespace, which would
     silently break COUNT's name-based matching against species names like
     "Homo_sapiens").
  2. Matrix: CSV (species rows x gene columns) -> TSV (gene rows x species
     columns). Some gene descriptions in the CSV contain commas inside
     quotes (e.g. "kinase, mitochondrial"), so this uses Python's built-in
     csv module -- which handles quoted commas correctly -- rather than a
     naive split on ",".
  3. Verifies the tree and matrix have exactly the same set of species.

Usage:
    python3 prepare_inputs.py TREE_NEXUS MATRIX_CSV OUT_NEWICK OUT_TABLE
"""
import csv
import re
import sys


def extract_newick(nexus_text):
    """Pull the bare tree statement out of a NEXUS document's
    'tree NAME = [&R](...);' line."""
    match = re.search(r"^\s*tree\s+\S+\s*=\s*(\[&[RU]\])?\s*(\(.*;)", nexus_text, re.MULTILINE)
    if not match:
        raise ValueError("No 'tree NAME = (...);' statement found in NEXUS file")
    return match.group(2)


def quote_species_names(newick_text):
    """Wrap every leaf label in single quotes so COUNT's Newick parser
    keeps underscores literal instead of treating them as whitespace."""
    return re.sub(r"([(,])([A-Za-z0-9_.-]+):", r"\1'\2':", newick_text)


def convert_tree(tree_nexus_path, out_newick_path):
    with open(tree_nexus_path) as f:
        nexus_text = f.read()
    newick = quote_species_names(extract_newick(nexus_text))
    with open(out_newick_path, "w") as f:
        f.write(newick + "\n")
    species = set(re.findall(r"'([A-Za-z0-9_.-]+)':", newick))
    return species


def convert_matrix(matrix_csv_path, out_table_path):
    with open(matrix_csv_path, newline="") as f:
        rows = list(csv.reader(f))

    genes = rows[0][1:]
    species = [row[0] for row in rows[1:]]

    with open(out_table_path, "w") as out:
        out.write("family\t" + "\t".join(species) + "\n")
        for col, gene in enumerate(genes, start=1):
            values = (row[col] for row in rows[1:])
            out.write(gene + "\t" + "\t".join(values) + "\n")

    return set(species), len(genes)


def main():
    tree_nexus, matrix_csv, out_newick, out_table = sys.argv[1:5]

    print("[Step 1/3] Converting NEXUS tree to quoted Newick format...")
    tree_species = convert_tree(tree_nexus, out_newick)
    print(f"  -> Saved: {out_newick} ({len(tree_species)} species in tree)")

    print("[Step 2/3] Transposing presence/absence matrix for COUNT...")
    table_species, n_genes = convert_matrix(matrix_csv, out_table)
    print(f"  -> Saved: {out_table} ({n_genes} genes x {len(table_species)} species)")

    print("[Step 3/3] Checking species alignment between tree and table...")
    missing_in_tree = table_species - tree_species
    missing_in_table = tree_species - table_species
    if missing_in_tree or missing_in_table:
        print(
            f"ERROR: Species mismatch! Matrix missing {len(missing_in_table)}, "
            f"Tree missing {len(missing_in_tree)}",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"  -> Verification passed: all {len(table_species)} species match perfectly.")


if __name__ == "__main__":
    main()
