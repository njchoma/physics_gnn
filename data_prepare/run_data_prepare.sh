#!/bin/bash
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -p debug
#SBATCH -J data_prepare
#SBATCH -t 20:00:00

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread


#run the application:
srun -n 1 -c 64 --cpu_bind=cores /global/homes/r/rochette/lhc_gnn/data_prepare/data_prepare.sh

