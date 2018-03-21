#!/bin/bash
#
# all commands that start with SBATCH contain commands that are just used by SLURM for scheduling
#################
# set a job name
#SBATCH --job-name=GCNN
#################
# a file for job output, you can check job progress
# SBATCH --output=GPUTFtest.out
#################
# a file for errors from the job
# SBATCH --error=GPUTFtest.err
#################
# time you think you need; default is one hour
# in minutes
# In this case, hh:mm:ss, select whatever time you want, the less you ask for the # faster your job will run.
# Default is one hour, this example will run in  less that 5 minutes.
#SBATCH --time=2-00:00:00
#################
# --gres will give you one GPU, you can ask for more, up to 4 (or how ever many are on the node/card)
#SBATCH --gres gpu:1
# We are submitting to the batch partition
#SBATCH --qos=batch
#################
#number of nodes you are requesting
#SBATCH --nodes=1
#################
#memory per node; default is 4000 MB per CPU
#SBATCH --mem=10000
#################
# Have SLURM send you an email when the job ends or fails, careful, the email could end up in your clutter folder
#SBATCH --mail-type=ALL # notifications for job done & fail
#SBATCH --mail-user=nc2201@courant.nyu.edu


NBTRAIN=100000
NBBATCH=100
NBFMAP=64
NBLAYER=12
# NBTEST=$NBTRAIN
NBTEST=10500
NBPRINT=$((($NBTRAIN/$NBBATCH)/10))
NBPRINT=$(($NBPRINT>1?$NBPRINT:1))
JOBNAME="lg_""QCDbaseline_smtest_""$NBFMAP""_""$NBLAYER""_""$SLURM_ARRAY_TASK_ID"
PYARGS="--name $JOBNAME --kernels QCDAwareMeanNorm --nb_batch $NBBATCH --combine_kernels Fixed_Balanced --data NYU --fm $NBFMAP --depth $NBLAYER --nb_MLPadj_hidden 4 --nbtrain $NBTRAIN --nbtest $NBTEST --nbprint $NBPRINT --lr 0.005 --lrdecay 0.9 --cuda"


# Network using a single kernel as described in "QCD-Aware Recursive Neural Networks for Jet Physics"
srun python3 script/main.py $PYARGS
