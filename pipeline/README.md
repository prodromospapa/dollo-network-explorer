# Simplified Dollo Co-Evolution Pipeline (Supervisor Guide)

This directory contains clean, simplified versions of the bash pipeline scripts designed for easy code walkthroughs and supervisor presentations.

Each numbered script is a short bash wrapper (usually under 30 lines) that
just points a helper script at the right files -- the actual logic (CSV
parsing, Jaccard scoring) lives in `helpers/*.py`. This keeps the bash
scripts readable as a plain sequence of steps, while the real computation
is ordinary, linear Python rather than awk one-liners.

**Scope of this version:** raw Dollo concordance scores only -- no
significance test / p-value yet (deliberately deferred, see the bottom of
this file). The output ranks candidate genes by how much their independent
loss history overlaps with the seed gene's, with no statement yet about
whether that overlap is more than chance.

---

## 1. What is Dollo Co-Evolution?

* **Dollo's Law:** A complex trait or gene arises **at most once** in evolutionary history (single origin), but can be lost independently across descendant species lineages.
* **Functional Co-Evolution:** If two genes function together in the same molecular pathway (e.g. primary cilia), evolutionary pressure to maintain them is coupled. When a lineage loses the organelle or pathway, both genes are lost on the **same phylogenetic branches**.
* **Goal of the Pipeline:** Discover genes functionally linked to a target query gene (e.g. `SCAPER` or `NPHP4`) by finding genes that share the exact same branch-loss events across 196 species.

---

## 2. Pipeline Overview

| Script | Purpose | What to Tell Your Supervisor |
| :--- | :--- | :--- |
| **`00_install_count.sh`** | Tool Download | Downloads the official [COUNT](https://github.com/csurosm/count) phylogenetic software release JAR. |
| **`01_prepare_inputs.sh`** | Format Alignment | Converts NEXUS tree to Newick with quoted taxa names, and transposes the binary matrix to gene rows $\times$ species columns. |
| **`02_reconstruct.sh`** | Dollo Reconstruction | Runs COUNT's parsimony with `-gain 1000 -loss 1` to reconstruct every gene's ancestral gain and loss branches. |
| **`03_concordance.sh`** | Jaccard Concordance | Calculates the loss-branch Jaccard similarity between the seed gene and all ~19,757 candidate genes, and writes the ranked table. |

---

## 3. Two Key Concepts to Explain

### A. Why `-gain 1000 -loss 1`?
* COUNT's command line provides weighted parsimony: $\text{Total Cost} = (\text{gain} \times N_{\text{gains}}) + (\text{loss} \times N_{\text{losses}})$.
* On our 196-species tree, there are **389 branches** in total.
* The maximum possible number of losses is 389.
* By setting `gain = 1000` (which is strictly $> 389$) and `loss = 1`, the algorithm will **never** choose to hypothesize a second gain to save on losses. This mathematically enforces Dollo parsimony (at most 1 gain).

### B. Why Jaccard Similarity on Loss Branches?
* Dollo parsimony assigns every gene a single gain, so gain timing reflects origin depth rather than co-evolution.
* Independent co-losses across tree branches provide the actual evolutionary coupling signal.
* We compute Jaccard similarity:
  $$\text{Jaccard} = \frac{|\text{Seed Losses} \cap \text{Candidate Losses}|}{|\text{Seed Losses} \cup \text{Candidate Losses}|}$$
* Dividing by the union normalizes for genes with many versus few losses.

---

## 4. How to Run

```bash
# 1. Download COUNT
bash scripts_simple/00_install_count.sh

# 2. Prepare inputs (formats tree and matrix)
bash scripts_simple/01_prepare_inputs.sh

# 3. Reconstruct ancestral states (run once for all ~19,758 genes)
bash scripts_simple/02_reconstruct.sh

# 4. Score concordance for a seed gene -- writes results/SEED_concordance.tsv
bash scripts_simple/03_concordance.sh SCAPER
```

---

## 5. Deferred: statistical significance

A raw concordance score has no attached probability -- two genes can look
concordant purely from shared phylogenetic inertia (e.g. both happening to
be absent from the same large non-ciliated clade), without real functional
coupling. A permutation-based significance test (shuffle which species
carry the candidate gene, holding its presence count fixed, re-reconstruct
with COUNT, and see how often a random gene scores as high as the real
one) plus Benjamini-Hochberg FDR correction across all tested candidates
was built and validated earlier, but is deliberately left out of this
version to keep the pipeline to its essential four steps. Treat
`results/*_concordance.tsv` as a **ranked screen**, not a list of
statistically confirmed hits, until that step is added back.
