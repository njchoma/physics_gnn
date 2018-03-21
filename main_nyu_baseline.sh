#!/bin/bash

#SBATCH --job-name=GCNN
# SBATCH --output=slurm_out/GCNN_%j_%a.out
# SBATCH --error=GPUTFtest.err
#SBATCH --time=2-00:00:00
#SBATCH --gres gpu:1
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --mem=10000
#SBATCH --mail-type=ALL # notifications for job done & fail
#SBATCH --mail-user=nc2201@courant.nyu.edu

#########
# DATASET
#########
DATASET="NYU"
NBTRAIN=100000
# NBTEST=$NBTRAIN
NBTEST=10500

##################
# MODEL PARAMETERS
##################
NBBATCH=100
NBFMAP=64
NBLAYER=6
LRATE=0.005
KERNELS="QCDAwareMeanNorm"
CMBKER="Fixed_Balanced"

JOBNAME="qcd_smTest_""$NBFMAP""_""$NBLAYER""_""$SLURM_ARRAY_TASK_ID"

################
# FIXED (mostly)
################
NBPRINT=$((($NBTRAIN/$NBBATCH)/10))
NBPRINT=$(($NBPRINT>1?$NBPRINT:1))
PYARGS="--name $JOBNAME --kernels $KERNELS --nb_batch $NBBATCH --combine_kernels $CMBKER --data $DATASET --fm $NBFMAP --depth $NBLAYER --nb_MLPadj_hidden 4 --nbtrain $NBTRAIN --nbtest $NBTEST --nbprint $NBPRINT --lr $LRATE --lrdecay 0.9 --cuda"


# Network using a single kernel as described in "QCD-Aware Recursive Neural Networks for Jet Physics"
srun python3 script/main.py $PYARGS
