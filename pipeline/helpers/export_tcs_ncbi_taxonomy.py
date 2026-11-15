#!/usr/bin/env python3
"""
Give the 196-species TCS dataset the same NCBI taxonomy ranks as the EggNOG
dataset, for the alignment viewer's taxonomy filter and colour-by options.

The TCS tree_layout.json carries a hand-curated scheme (supergroup / kingdom /
phylum / detailed / tcs). EggNOG's viewer uses NCBI ranks instead
(domain, kingdom, phylum, class, order, family, genus), built by
export_eggnog_circular_layout.py. This script resolves each TCS species name
against the local NCBI taxonomy (ete3) and writes the same rank names, using
the same leaf_taxonomy() rules (fall back to the nearest higher NCBI rank).

Species names that NCBI does not know verbatim (e.g. 'Blastocystis_sp_subtype4-...')
are resolved through their genus; if that fails they are 'Unclassified', as in EggNOG.

Output (repo root): tcs_taxonomy_ncbi.json
  {"tax_levels": [...], "species": {"<Species_name>": {"sci_name": ..., "domain": ..., ...}}}

Usage (from repo root):
  python3 pipeline/helpers/export_tcs_ncbi_taxonomy.py
"""
import json
import sys
from pathlib import Path

from ete3 import NCBITaxa

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_eggnog_circular_layout import TAX_LEVELS, leaf_taxonomy  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SPECIES_TXT = ROOT / "data" / "tcs_species.txt"
OUT = ROOT / "tcs_taxonomy_ncbi.json"


def resolve_taxids(ncbi, species):
    """species name -> NCBI taxid string (verbatim name, else its genus)."""
    out = {}
    for sp in species:
        name = sp.replace("_", " ")
        hit = ncbi.get_name_translator([name]).get(name)
        if not hit:
            genus = name.split(" ")[0]
            hit = ncbi.get_name_translator([genus]).get(genus)
            if hit:
                print(f"{sp}: not in NCBI as named, using genus {genus}")
        if hit:
            out[sp] = str(hit[0])
        else:
            print(f"{sp}: not found in NCBI, Unclassified")
    return out


def main():
    species = [s.strip() for s in SPECIES_TXT.read_text().splitlines() if s.strip()]
    ncbi = NCBITaxa()
    taxids = resolve_taxids(ncbi, species)
    tax = leaf_taxonomy(ncbi, list(taxids.values()))
    result = {}
    for sp in species:
        if sp in taxids:
            result[sp] = tax[taxids[sp]]
        else:
            result[sp] = {"sci_name": sp.replace("_", " "), **{lvl: "Unclassified" for lvl in TAX_LEVELS}}
    OUT.write_text(json.dumps({"tax_levels": TAX_LEVELS, "species": result}, indent=1))
    print(f"Wrote {OUT} for {len(result)} species")


if __name__ == "__main__":
    main()
