"""Unit tests for pipeline/helpers/{concordance,prepare_inputs,reconstruct}.py."""
import importlib.util
import sys
from pathlib import Path

import pytest

HELPERS_DIR = Path(__file__).resolve().parent.parent / "pipeline" / "helpers"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HELPERS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


concordance = _load("concordance")
prepare_inputs = _load("prepare_inputs")
reconstruct = _load("reconstruct")


# --- concordance.py ---------------------------------------------------

def _write_events(path, rows):
    with open(path, "w") as f:
        f.write("gene\tnodeidx\tpresence\tgain\tloss\n")
        for row in rows:
            f.write("\t".join(row) + "\n")


def test_load_loss_branches_preserves_first_seen_order(tmp_path):
    events = tmp_path / "events.tsv"
    _write_events(events, [
        ("GENE_B", "1", "0", "0", "1"),
        ("GENE_A", "2", "0", "0", "1"),
        ("GENE_B", "3", "0", "0", "1"),
    ])
    all_genes, _ = concordance.load_loss_branches(str(events))
    assert all_genes == ["GENE_B", "GENE_A"]


def test_load_loss_branches_zero_loss_gene_present_with_no_entry(tmp_path):
    events = tmp_path / "events.tsv"
    _write_events(events, [
        ("GENE_UNIVERSAL", "1", "1", "1", "0"),
        ("GENE_UNIVERSAL", "2", "1", "0", "0"),
    ])
    all_genes, losses = concordance.load_loss_branches(str(events))
    assert all_genes == ["GENE_UNIVERSAL"]
    assert losses.get("GENE_UNIVERSAL", set()) == set()


def test_load_loss_branches_accumulates_multiple_loss_rows(tmp_path):
    events = tmp_path / "events.tsv"
    _write_events(events, [
        ("GENE_A", "1", "0", "0", "1"),
        ("GENE_A", "2", "0", "0", "1"),
        ("GENE_A", "3", "0", "1", "0"),
    ])
    _, losses = concordance.load_loss_branches(str(events))
    assert losses["GENE_A"] == {"1", "2"}


def test_load_loss_branches_interleaved_gene_rows_tracked_as_one_gene(tmp_path):
    # Mirrors the documented file-order-independence bug in tests/run_tests.sh:
    # rows for one gene can appear both before and after another gene's rows.
    events = tmp_path / "events.tsv"
    _write_events(events, [
        ("before_seed_gene", "1", "0", "0", "1"),
        ("seedgene", "1", "0", "0", "1"),
        ("before_seed_gene", "2", "0", "0", "1"),
        ("seedgene", "2", "0", "0", "1"),
    ])
    all_genes, losses = concordance.load_loss_branches(str(events))
    assert all_genes == ["before_seed_gene", "seedgene"]
    assert losses["before_seed_gene"] == {"1", "2"}
    assert losses["seedgene"] == {"1", "2"}


def _write_csv(path, header, rows):
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_read_gene_set_collects_nonempty_stripped_symbols(tmp_path):
    csv_path = tmp_path / "genes.csv"
    _write_csv(csv_path, ["resolved_symbol", "other"], [
        {"resolved_symbol": " SCAPER ", "other": "x"},
        {"resolved_symbol": "NPHP4", "other": "y"},
    ])
    genes = concordance.read_gene_set(str(csv_path), "resolved_symbol")
    assert genes == {"SCAPER", "NPHP4"}


def test_read_gene_set_skips_blank_and_whitespace_only_values(tmp_path):
    csv_path = tmp_path / "genes.csv"
    _write_csv(csv_path, ["resolved_symbol"], [
        {"resolved_symbol": ""},
        {"resolved_symbol": "   "},
        {"resolved_symbol": "REAL_GENE"},
    ])
    genes = concordance.read_gene_set(str(csv_path), "resolved_symbol")
    assert genes == {"REAL_GENE"}


def test_read_gene_set_missing_column_key_does_not_crash(tmp_path):
    csv_path = tmp_path / "genes.csv"
    # Row has an extra unnamed field, DictReader stores it under None key,
    # but requesting a genuinely absent column name must not raise.
    with open(csv_path, "w") as f:
        f.write("some_other_column\n")
        f.write("value\n")
    genes = concordance.read_gene_set(str(csv_path), "resolved_symbol")
    assert genes == set()


def test_score_candidates_jaccard_and_zero_union_is_zero_not_error():
    all_genes = ["seed", "candidate_partial", "candidate_zero"]
    losses_by_gene = {
        "seed": {"n1", "n2"},
        "candidate_partial": {"n2", "n3"},
        # candidate_zero has no entry at all -> union with seed is nonzero via seed's own losses
    }
    rows = concordance.score_candidates(all_genes, losses_by_gene, "seed")
    by_gene = {r[0]: r for r in rows}

    partial = by_gene["candidate_partial"]
    # shared={n2}=1, union={n1,n2,n3}=3
    assert partial[2] == 1
    assert partial[3] == 3
    assert partial[4] == pytest.approx(1 / 3)

    zero = by_gene["candidate_zero"]
    assert zero[3] == 2  # union is seed's own losses
    assert zero[4] == 0.0


def test_score_candidates_zero_union_when_both_empty():
    all_genes = ["seed", "candidate"]
    losses_by_gene = {"seed": set(), "candidate": set()}
    rows = concordance.score_candidates(all_genes, losses_by_gene, "seed")
    assert rows == [("candidate", 0, 0, 0, 0.0)]


def test_score_candidates_excludes_seed_gene_itself():
    all_genes = ["seed", "other"]
    losses_by_gene = {"seed": {"n1"}, "other": {"n1"}}
    rows = concordance.score_candidates(all_genes, losses_by_gene, "seed")
    assert [r[0] for r in rows] == ["other"]


def test_score_candidates_sort_descending_jaccard_ties_alphabetical():
    all_genes = ["seed", "zeta", "alpha", "beta"]
    # zeta and alpha both share all losses with seed (jaccard=1.0); beta shares none.
    losses_by_gene = {
        "seed": {"n1"},
        "zeta": {"n1"},
        "alpha": {"n1"},
        "beta": {"n2"},
    }
    rows = concordance.score_candidates(all_genes, losses_by_gene, "seed")
    assert [r[0] for r in rows] == ["alpha", "zeta", "beta"]


# --- prepare_inputs.py --------------------------------------------------

def test_extract_newick_with_rooted_tag():
    nexus = "begin trees;\ntree TREE1 = [&R] (A:1,B:1);\nend;\n"
    assert prepare_inputs.extract_newick(nexus) == "(A:1,B:1);"


def test_extract_newick_with_unrooted_tag():
    nexus = "tree TREE1 = [&U] (A:1,B:1);\n"
    assert prepare_inputs.extract_newick(nexus) == "(A:1,B:1);"


def test_extract_newick_with_no_rooting_tag():
    nexus = "tree TREE1 = (A:1,B:1);\n"
    assert prepare_inputs.extract_newick(nexus) == "(A:1,B:1);"


def test_extract_newick_raises_when_no_tree_statement():
    with pytest.raises(ValueError):
        prepare_inputs.extract_newick("begin trees;\nend;\n")


def test_quote_species_names_wraps_underscore_names():
    newick = "(Homo_sapiens:1,Mus_musculus:2);"
    quoted = prepare_inputs.quote_species_names(newick)
    assert quoted == "('Homo_sapiens':1,'Mus_musculus':2);"


def test_quote_species_names_leaves_structure_alone():
    newick = "((A:1,B:2):1,C:3);"
    quoted = prepare_inputs.quote_species_names(newick)
    assert quoted == "(('A':1,'B':2):1,'C':3);"


def test_convert_tree_end_to_end(tmp_path):
    nexus_path = tmp_path / "tree.nex"
    out_newick = tmp_path / "out.newick"
    nexus_path.write_text("tree TREE1 = [&R] (Homo_sapiens:1,Mus_musculus:2);\n")

    species = prepare_inputs.convert_tree(str(nexus_path), str(out_newick))

    assert species == {"Homo_sapiens", "Mus_musculus"}
    assert out_newick.read_text() == "('Homo_sapiens':1,'Mus_musculus':2);\n"


def test_convert_matrix_end_to_end_handles_quoted_comma(tmp_path):
    csv_path = tmp_path / "matrix.csv"
    out_table = tmp_path / "out.tsv"
    csv_path.write_text(
        'species,GENE1,"GENE2, kinase"\n'
        "Homo_sapiens,1,0\n"
        "Mus_musculus,0,1\n"
    )

    species, n_genes = prepare_inputs.convert_matrix(str(csv_path), str(out_table))

    assert species == {"Homo_sapiens", "Mus_musculus"}
    assert n_genes == 2
    lines = out_table.read_text().splitlines()
    assert lines[0] == "family\tHomo_sapiens\tMus_musculus"
    assert lines[1] == "GENE1\t1\t0"
    assert lines[2] == "GENE2, kinase\t0\t1"


# --- reconstruct.py ------------------------------------------------------

def test_load_gene_names_skips_header_and_splits_on_first_tab_only(tmp_path):
    table_path = tmp_path / "table.tsv"
    table_path.write_text(
        "family\tSpeciesA\tSpeciesB\n"
        "GENE1\t1\t0\n"
        "GENE2\t0\t1\n"
    )
    names = reconstruct.load_gene_names(str(table_path))
    assert names == ["GENE1", "GENE2"]


def test_format_events_maps_family_idx_and_skips_non_history_and_header_rows(tmp_path):
    table_path = tmp_path / "table.tsv"
    table_path.write_text(
        "family\tSpeciesA\tSpeciesB\n"
        "GENE0\t1\t0\n"
        "GENE1\t0\t1\n"
    )
    raw_path = tmp_path / "raw.tsv"
    # COUNT's raw stdout: HISTORY rows, a non-HISTORY row to skip, and a
    # 'HISTORY family ...' header-echo row that must also be skipped.
    raw_path.write_text(
        "HISTORY\tfamily\tnodeidx\tpresence\tmulti\tmaxpresence\tgain\tloss\n"
        "SOMETHINGELSE\t0\t1\t1\t0\t1\t0\t0\n"
        "HISTORY\t0\t1\t1\t0\t1\t1\t0\n"
        "HISTORY\t1\t2\t1\t0\t1\t0\t1\n"
    )
    events_path = tmp_path / "events.tsv"

    n_rows = reconstruct.format_events(str(table_path), str(raw_path), str(events_path))

    assert n_rows == 2
    lines = events_path.read_text().splitlines()
    assert lines[0] == "gene\tnodeidx\tpresence\tgain\tloss"
    assert lines[1] == "GENE0\t1\t1\t1\t0"
    assert lines[2] == "GENE1\t2\t1\t0\t1"


def test_spot_check_prints_summed_gain_loss_per_gene(tmp_path, capsys):
    events_path = tmp_path / "events.tsv"
    events_path.write_text(
        "gene\tnodeidx\tpresence\tgain\tloss\n"
        "SCAPER\t1\t1\t1\t0\n"
        "SCAPER\t2\t0\t0\t1\n"
        "NPHP4\t1\t1\t1\t0\n"
    )

    reconstruct.spot_check(str(events_path), ["SCAPER", "NPHP4"])

    out = capsys.readouterr().out
    assert "SCAPER: gain=1, loss=1" in out
    assert "NPHP4: gain=1, loss=0" in out


def test_spot_check_does_not_crash_for_gene_with_no_matching_rows(tmp_path, capsys):
    events_path = tmp_path / "events.tsv"
    events_path.write_text(
        "gene\tnodeidx\tpresence\tgain\tloss\n"
        "SCAPER\t1\t1\t1\t0\n"
    )

    reconstruct.spot_check(str(events_path), ["SCAPER", "GENE_NOT_PRESENT"])

    out = capsys.readouterr().out
    assert "SCAPER: gain=1, loss=0" in out
    assert "GENE_NOT_PRESENT: gain=0, loss=0" in out
