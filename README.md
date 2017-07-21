# GCNN

### graph-convolutions based neural networks and a few applications 

## Code organization
This repository is organized as such :
```
GCNN/
    |-- README.md
    |-- main.sh
    |-- script/  # python code
    |   |-- main.py            # main file : reads arguments, call main_nyu or main_nersc to train and test
    |   |-- main_nyu.py        # train & test with projectNYU on data from dataNYU
    |   |-- main_nersc.py      # train & test with projectNERSC on data from dataNERSC
    |	|-- main_multiprocess  # runs the main file on a list of arguments contained in GCNN/args.txt
    |	|-- read_args.py       # functions related to argument reading and network initialization
    |	|-- model/             # code related to graph convolutions, kernels and networks
    |	|-- utils/             # small function usefull in different files
    |	|-- graphics/          # code for plots
    |	|-- projectNERSC/      # code specific to training/testing on the NERSC data
    |	|-- projectNYU/        # code specific to training/testing on the NYU data
    |
    |-- paramNYU.txt    # file created when launching 'main.sh', contains paths to training and testing NYU data
    |-- dataNYU     # link to a directory containing NYU data
    |	|-- train_uncropped.pickle
    |	|-- train_cropped.pickle
    |	|-- test_cropped.pickle
    |-- dataNERSC   # link to a directory containing NERSC data
    |	|-- DelphesNevents
    |   |-- *01.h5  # testing set (multiple files)
    |   |-- *02.h5  # training set (multiple files)
    |
    |-- modelsNYU    # directory where trained networks for NYU data are saved
    |-- modelsNERSC  # directory where trained networks for NERSC data are saved
    |
    |-- args.txt  # sequence of arguments for multiprocessing : each line will be used as an independant set of arguments
```

## Before training a network

`main.sh` contains commented out command lines that launch training on models with different architecture. You can change parameters used to initialize networks, and more importantly you should add `--data NERSC` if you are not running on the default NYU data.

Before training a network, create a link `dataNYU` or `dataNERSC` (depending on the data you want to train on) and run `main.sh` once, specifying the data you want to use : it will create the corresponding `param{}.txt` file and assume default paths. You can modify the path to fit a different organization of your local files. If your file system is the same as the default one, this will simply train on the architecture selected in `main.sh`.

## Training a network

To train a network, you simply need to launch `main.sh` after adding a command line defining the network you want to train, or use a single command calling `python script/main.py [arguments]`. Run `python script/main.py --help` to see a list of arguments.

The main arguments used to design a network are :
* `--kernel str` : type of kernel / architecture used.\
* `--fm int` : number of node feature maps at each layer
* `--edge_fm int` : number of edge feature maps at each layer (used only with multikernels)
* `--depth int` : number of layers in the network

You can also specify the number of events used every epoch for training, testing, the frequency at which results will be printed, the learning rate and its rate of decay. Train on GPU with `--cuda`.

The network will be saved every epoch, and some stats will be saved in a csv file.

## NYU Data

## NERSC Data

NERSC data is handle the way it is stored on the NERSC server.

##### Testing and Training sets
The current code uses files ending in "01.h5" as testing set and those ending in "02.h5" as training set. This can be changed in the code : `script/projectNERSC/model_raw_nersc.py` contains two functions `is_test` and `is_train` taking a file's name as argument, changing those functions result in changing the files used for each set. Be careful when changing those functions not to use files for both sets.

##### Event selection for one epoch
An epoch is performed on N events by randomly (uniformly) selecting N events in all relevant files. This selection is done at each epoch.

##### Data structure details
Here are the data structure assumed and the fields used in those hdf5 files :
```
file.h5  # files starting with "GG" are class 1, others are class 0
    |-- event_0
    |   |-- clusE    # Energy of each particle detected
    |   |-- clusEta  # [Eta wiki](https://en.m.wikipedia.org/wiki/Pseudorapidity)
    |   |-- clusPhi  # [Phi wiki](https://en.m.wikipedia.org/wiki/Azimuth)
    |   |-- clusEM   # Electromagnetic energy
    |
    |-- event_1
    |   |...
    |...
    |-- event_`nb_event_in_file`
```
Those hdf5 are also assumed to have an attribute `nb_event` : `hdf5file['event_idx'].attrs['nb_event']` should return the number of events in the file loaded as `hdf5file = h5py.File(path_to_file, 'r')`.

##### DelphesNevents
This file contains a number for each type of hdf5 data file, which is used to compute the weights used for training and testing. 

