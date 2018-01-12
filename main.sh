#!/bin/bash

# This script contains examples of command lines to train a network.
# Uncomment a line to use it.

# Network using a single kernel as described in "QCD-Aware Recursive Neural Networks for Jet Physics"
python3 script/main.py --name test_mpnn --kernel QCDAwareMeanNorm --adj_kernel MPNNdirected --data NYU --fm 16 --depth 4 --nbtrain 1000 --nbtest 1000 --nbprint 50 --lr 0.001 --lrdecay 0.9 --cuda

# Other Networks
# pass

# ########## Still being tested : ########## #

# Network using multiple kernels as described in "QCD-Aware Recursive Neural Networks for Jet Physics". Those kernels are the same for all layers.
# python3 script/main.py --name multi_100ktrain_24.4.8 --kernel MultiQCDAware --data NYU --fm 24 --edge_fm 4 --depth 8 --nbtrain 100000 --nbtest 50000 --nbprint 5000 --lr 0.005 --lrdecay 0.90 --cuda

