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
    |		|-- train_uncropped.pickle
    |		|-- train_cropped.pickle
    |		|-- test_cropped.pickle
    |-- dataNERSC  # link to a directory containing NERSC data
    |		|-- train.h5
    |		|-- test.h5
    |
    |-- modelsNYU  # directory where trained networks for NYU data are saved
    |-- modelsNERSC  # directory where trained networks for NERSC data are saved
    |
    |-- paramNYU.txt  # file created when launching 'main.sh', contains paths to training and testing NYU data
    |-- paramNERSC.txt  # file created when launching 'main.sh', contains paths to training and testing NERSC data
    |
    |-- args.txt  # sequence of arguments for multiprocessing : each line will be used as an independant set of arguments
```

### Before training a network

`main.sh` contains commented out command lines that launch training on models with different architecture. You can change parameters used to initialize networks, and more importantly you should add `--data NERSC` if you are not running on the default NYU data.

Before training a network, create a link `dataNYU` or `dataNERSC` (depending on the data you want to train on) and run `main.sh` once, specifying the data you want to use : it will create the corresponding `param{}.txt` file and assume default paths. You can modify the path to fit a different organization of your local files. If your file system is the same as the default one, this will simply train on the architecture selected in `main.sh`.

### Training a network

To train a network, you simply need to launch `main.sh` after adding a command line defining the network you want to train, or use a single command calling `python script/main.py [arguments]`. Run `python script/main.py --help` to see a list of arguments.

The main arguments used to design a network are :
* `--kernel str` : type of kernel / architecture used.\
* `--fm int` : number of node feature maps at each layer
* `--edge_fm int` : number of edge feature maps at each layer (used only with multikernels)
* `--depth int` : number of layers in the network

You can also specify the number of events used every epoch for training, testing, the frequency at which results will be printed, the learning rate and its rate of decay. Train on GPU with `--cuda`.

The network will be saved every epoch, and some stats will be saved in a csv file.
