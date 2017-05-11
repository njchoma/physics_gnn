#!/bin/bash

module purge
module load python
module load deeplearning

time python data_prepare.py --rawdata $HOME'/rawdata' --data $HOME'/lhc_gnn/data' --stdout advance.out
echo 'DONE\n'