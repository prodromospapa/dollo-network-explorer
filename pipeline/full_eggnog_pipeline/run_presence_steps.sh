#!/usr/bin/env bash
# Website build for the full EggNOG dataset, after run_final_steps.sh.
#   - Dollo (COUNT) co-loss  -> Leiden modules (graph views), loss counts
#   - Dollo (COUNT) history  -> where genes were gained/lost on the tree
#   - presence/absence Jaccard -> partner scores / distances
set -euo pipefail
cd "$(dirname "$0")/../.."

echo "[1/3] Gene list, Leiden modules, cilia annotations -> data_eggnog.json..."
python3 pipeline/N6_build_html_explorer_filtered.py --dataset eggnog --output /dev/null
cp pipeline/data_eggnog.json data_eggnog.json

echo "[2/3] Circular tree layout, presence bits, Dollo loss/gain events..."
python3 pipeline/helpers/export_eggnog_circular_layout.py

echo "[3/3] Presence/absence Jaccard partner lists -> network_partners_eggnog.bin..."
python3 pipeline/helpers/build_presence_partners.py



# Cache-bust: browsers must not reuse old data files with the new page
sed -i "s/^const DATA_VERSION = '[^']*';/const DATA_VERSION = '$(date +%Y%m%d-%H%M)';/" eggnog.html
echo "Done."
