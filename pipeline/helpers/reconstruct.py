#!/usr/bin/env python3
"""Helper for 02_reconstruct.sh.

Formats COUNT's raw reconstruction output into a clean events table.

COUNT's raw stdout has one line per (gene, tree-node) pair, tagged
"HISTORY", with the gene identified only by its row position in the input
table (an integer index), not its name:

    HISTORY  family_idx  nodeidx  presence  multi  maxpresence  gain  loss  ...

This maps family_idx back to the real gene name (from the same table.tsv
row order COUNT read) and keeps only the columns needed downstream: gene,
nodeidx, presence, gain, loss.

Usage:
    python3 reconstruct.py TABLE_TSV RAW_COUNT_OUTPUT EVENTS_OUT
"""
import sys


def load_gene_names(table_path):
    """Gene names in the same order COUNT read them (table.tsv's rows,
    skipping the header) -- COUNT's family_idx is this row's position."""
    with open(table_path) as f:
        next(f)  # header
        return [line.split("\t", 1)[0] for line in f]


def format_events(table_path, raw_path, events_path):
    gene_names = load_gene_names(table_path)

    with open(raw_path) as raw, open(events_path, "w") as out:
        out.write("gene\tnodeidx\tpresence\tgain\tloss\n")
        n_rows = 0
        for line in raw:
            fields = line.rstrip("\n").split("\t")
            if fields[0] != "HISTORY" or fields[1] == "family":
                continue
            family_idx = int(fields[1])
            gene = gene_names[family_idx]
            nodeidx, presence, gain, loss = fields[2], fields[3], fields[6], fields[7]
            out.write(f"{gene}\t{nodeidx}\t{presence}\t{gain}\t{loss}\n")
            n_rows += 1
    return n_rows


def spot_check(events_path, genes):
    totals = {g: [0, 0] for g in genes}  # gene -> [gain, loss]
    with open(events_path) as f:
        next(f)
        for line in f:
            gene, _, _, gain, loss = line.rstrip("\n").split("\t")
            if gene in totals:
                totals[gene][0] += int(gain)
                totals[gene][1] += int(loss)
    for gene in genes:
        gain, loss = totals[gene]
        print(f"  {gene}: gain={gain}, loss={loss}")


def main():
    table_path, raw_path, events_path = sys.argv[1:4]

    print("[Step 2/2] Formatting events table...")
    n_rows = format_events(table_path, raw_path, events_path)
    print(f"  -> Saved: {events_path} ({n_rows} gene-node records)")

    print()
    print("Spot-check results:")
    spot_check(events_path, ["SCAPER", "NPHP4"])


if __name__ == "__main__":
    main()
