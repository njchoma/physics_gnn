#!/bin/bash

# This script contains examples of command lines to train a network.
# Uncomment a line to use it.

# Network using a single kernel as described in "QCD-Aware Recursive Neural Networks for Jet Physics"
python script/main.py --name test_QCDAwareMeanNorm --kernel QCDAwareMeanNorm --data NYU --fm 64 --depth 12 --nbtrain 1000000 --nbtest 1000000 --nbprint 20000 --lr 0.001 --lrdecay 0.9

# Other Networks
# pass

# ########## Still being tested : ########## #

# Network using multiple kernels as described in "QCD-Aware Recursive Neural Networks for Jet Physics". Those kernels are the same for all layers.
# python script/main.py --name test_LayerQCDAware --kernel LayerQCDAware --data NYU --fm 24 --edge_fm 4 --depth 8 --nbtrain 100000 --nbtest 5000 --nbprint 5000 --lr 0.03 --lrdecay 0.88

