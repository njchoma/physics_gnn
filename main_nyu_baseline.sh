#!/bin/bash

#SBATCH --job-name=GCNN
#SBATCH --output=slurm_out/GCNN_%A_%a.out
# SBATCH --error=GPUTFtest.err
#SBATCH --time=2-00:00:00
#SBATCH --gres gpu:1
#SBATCH --constraint=gpu_12gb
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --mem=10000
#SBATCH --mail-type=FAIL # notifications for job done & fail
#SBATCH --mail-user=nc2201@courant.nyu.edu

#########
# DATASET
#########
DATASET="NYU"
NBTRAIN=100000
NBTEST=$NBTRAIN
# NBTEST=10500

##################
# MODEL PARAMETERS
##################
NBBATCH=100
NBFMAP=96
NBLAYER=8
LRATE=0.005
LRDECAY=0.96
OPTIONS="--nbepoch 100 --conv_type ResGNN --node_type Identity --readout Sum --cuda"
NBEXTRANODES=30

###################
# KERNEL PARAMETERS
###################
KERNELS="QCDAwareMeanNorm"
CMBKER="Fixed_Balanced"
NBHIDDEN=4 # only applies with MLPdirected kernel

JOBNAME="qcd_""$NBEXTRANODES""_""$LRATE""_""$LRDECAY""_""$NBFMAP""_""$NBLAYER""_""$SLURM_ARRAY_TASK_ID"

#############
# FIXED INPUT
#############
NBPRINT=$((($NBTRAIN/$NBBATCH)/10))
NBPRINT=$(($NBPRINT>1?$NBPRINT:1))
PYARGS="--name $JOBNAME --kernels $KERNELS --nb_batch $NBBATCH --combine_kernels $CMBKER --data $DATASET --fm $NBFMAP --depth $NBLAYER --nb_MLPadj_hidden $NBHIDDEN --nbtrain $NBTRAIN --nbtest $NBTEST --nbprint $NBPRINT --lr $LRATE --lrdecay $LRDECAY $OPTIONS --nb_extra_nodes $NBEXTRANODES"


# Network using a single kernel as described in "QCD-Aware Recursive Neural Networks for Jet Physics"
python3 script/main.py $PYARGS
