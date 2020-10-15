#!/bin/bash
#SBATCH -p scarf
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 30:00:00
#BSUB -o %j.log
#BSUB -e %j.err
python2 generate_goodenough_resolution.py $SLURM_ARRAY_TASK_ID
