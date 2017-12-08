#!/bin/bash

# This script contains examples of command lines to train a network.
# Uncomment a line to use it.

# Network using a single kernel as described in "QCD-Aware Recursive Neural Networks for Jet Physics"
python3 script/main.py --name ch_FQCDAware_lr0.005 --kernel FQCDAware --data NYU --fm 64 --depth 12 --nbtrain 200 --nbtest 100 --nbprint 50 --lr 0.005 --lrdecay 0.9 --cuda

# Other Networks
# pass

# ########## Still being tested : ########## #

# Network using multiple kernels as described in "QCD-Aware Recursive Neural Networks for Jet Physics". Those kernels are the same for all layers.
# python3 script/main.py --name test_LayerQCDAware --kernel LayerQCDAware --data NERSC --fm 24 --edge_fm 4 --depth 8 --nbtrain 10 --nbtest 5 --nbprint 2 --lr 0.03 --lrdecay 0.88

