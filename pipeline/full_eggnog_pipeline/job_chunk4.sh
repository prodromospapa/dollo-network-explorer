#!/bin/bash
#$ -cwd
cd /home/prodromosp/dollo-network-explorer

WORKERS=${NSLOTS:-12}
echo "Running chunk 4 with ${WORKERS} workers"

python3 pipeline/full_eggnog_pipeline/05_hmmsearch_filter_chunk.py --chunk 4 --total-chunks 5 --workers ${WORKERS}
