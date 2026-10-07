#!/bin/bash
# Full EggNOG pipeline: resume from step 3 (HMM filter) through step 8.
# Resource-capped for a shared cluster: everything is limited to the SGE slot
# reservation below (NSLOTS), and runs at low CPU/IO priority.
#
# Submit:   qsub pipeline/full_eggnog_pipeline/job_full_pipeline.sh
# Monitor:  qstat -u $USER ; tail -f pipeline/full_eggnog_pipeline/job_full_pipeline.log
# Change cores: edit the "-pe smp" line (everything else follows NSLOTS).
#$ -N eggnog_full
#$ -q all.q@narrativum.umcn.nl
#$ -pe smp 40
#$ -cwd
#$ -j y
#$ -o pipeline/full_eggnog_pipeline/job_full_pipeline.log
#$ -S /bin/bash
set -euo pipefail

export PATH=/home/prodromosp/miniconda3/bin:$PATH
N=${NSLOTS:-40}
# Stop numpy/BLAS/OpenMP/Java from grabbing all 128 host cores.
export OMP_NUM_THREADS=$N OPENBLAS_NUM_THREADS=$N MKL_NUM_THREADS=$N NUMEXPR_NUM_THREADS=$N
export COUNT_WORKERS=$N
# Lowest CPU + idle-class IO priority so interactive users are never slowed down.
renice -n 19 -p $$ >/dev/null
ionice -c 3 -p $$ 2>/dev/null || true

echo "=== eggnog_full on $(hostname), $N slots, started $(date)"
echo "[3/8] PANTHER HMM filter (resume)"
python3 pipeline/full_eggnog_pipeline/05_hmmsearch_filter_resume.py --workers "$N"

bash pipeline/full_eggnog_pipeline/run_final_steps.sh
echo "=== eggnog_full finished $(date)"
