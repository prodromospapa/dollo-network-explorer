import pandas as pd
from pathlib import Path

OUT_DIR = Path("data/eggnog_dataset")
CACHE_DIR = Path("cache_eggnog")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(OUT_DIR / "eggnog_presence.csv", index_col=0)

# Format for COUNT table.tsv
# First column should be "Family", followed by taxids
df.index.name = "Family"
df.to_csv(CACHE_DIR / "table.tsv", sep="\t")

import shutil
shutil.copy(OUT_DIR / "eggnog_tree.newick", CACHE_DIR / "tree.newick")

print("Created cache_eggnog/table.tsv and cache_eggnog/tree.newick")
