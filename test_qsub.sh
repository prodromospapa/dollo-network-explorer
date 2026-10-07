#!/bin/bash
#$ -q all.q@noggo.umcn.nl
#$ -pe smp 1
#$ -cwd
#$ -o test_qsub.log
#$ -e test_qsub.err
hostname
