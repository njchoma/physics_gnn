#!/bin/bash

module purge
module load python
module load deeplearning

time python preparedata.py --rawdata $HOME'/rawdata' --data $HOME'/lhc_gnn/data' --stdout advance.out --dotrain --dotest --wftrain --wftest --l2nntrain --l2nntest
echo 'DONE\n'