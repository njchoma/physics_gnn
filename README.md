# GCNN

#### graph-convolutions based neural networks and a few applications 

### Code organization
This repository is organized as such :
```
GCNN/
    |-- README.md
    |-- main.sh
    |-- script/  # python code
    |   	|-- main.py  # main file : read arguments and calls training and testing functions.
    |		|-- main_multiprocess  # runs the main file on a list of arguments contained in GCNN/args.txt
    |		|-- read_args.py  # functions related to argument reading and network initialization
    |		|-- model/  # code related to graph convolutions, kernels and networks
    |		|-- utils/  # small function usefull in different files
    |		|-- graphics/  # code for plots
    |		|-- projectNERSC/  # code specific to training/testing on the NERSC data
    |		|-- projectNYU/  # code specific to training/testing on the NYU data
    |
    |-- dataNYU  # link to a directory containing NYU data
    |-- dataNERSC  # link to a directory containing NERSC data
    |
    |-- modelsNYU  # directory where trained networks for NYU data are saved
    |-- modelsNERSC  # directory where trained networks for NERSC data are saved
    |
    |-- paramNYU.txt  # file created when launching 'main.sh', contains paths to training and testing NYU data
    |-- paramNERSC.txt  # file created when launching 'main.sh', contains paths to training and testing NERSC data
```
