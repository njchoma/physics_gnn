# GCNN

###  Graph neural network based models with applications in particle physics

## Code organization
This repository is organized as such :
```
GCNN/
    |-- main.sh            # stores model arguments and calls main.py
    |-- models[dataset]    # directory where trained networks for {NYU, NERSC, ICECUBE} data are saved
    |-- README.md
    |-- summarize.sh       # summarizes trained batch array models
    |-- script/              # python code
    | |-- main.py            # read arguments, load data from specified dataset, begin experiment
    | |-- experiment_handler.py # trains, tests over each epoch; perform training plots and save scores
    | |-- train_model.py     # train, test model over one epoch
    |	|-- data_ops/          # currently just used for generating and zero-padding minibatches
    |	|-- graphics/          # code for ROC AUC, 1/FPR scoring and all plots
    |	|-- loading/           
    |	| |-- data/              # load handling for all datasets
    |	| |-- model/             # all code for reading arguments, model save / restore, model global argument handling
    |	|-- model/             # contains model architectures and kernels
    |	|-- utils/             # small functions useful in different files
```

## Before training a network

`main.sh` contains commented out command lines that launch training on models with different architecture. You can change parameters used to initialize networks, and select the dataset using `--data {NYU, NERSC, ICECUBE}`. `--cuda` runs the network on a GPU.

## Training a network

To train a network, launch `main.sh` with argumented defined as desired. Run `python3 script/main.py --help` to see a list of arguments.

### Experiment arguments
* `--name str` : network reference name
* `--data str` : dataset to use. Choose between {NYU, ICECUBE, NERSC}
* `--cuda` : flag to train on GPU
* `--nbtrain int` : number of training samples to use
* `--nbtest int` : number of testing samples to use
* `--nbprint int` : print frequency by batch (note that at this point changing minibatch size changes print frequency
* `--quiet` : flag to reduce printing
* `--plot str` : type of plotting to perform. Choose from {spectral, spectral3d, eig, ker}
* `--save_best_model` : flag to save best model based on test 1/FPR
* `--tpr_target float` : set the TPR against which 1/FPR will be measures. Default is 0.5
* `--no_shuffle` : flag to load and run samples in the same order. Good for plotting
* `--nb_batch int` : minibatch size
* `--sorted_training` : flag to group similar-sized training samples (test does this by default). Minibatches of different sizes are padded with zeros so setting this flag significantly speeds up training. However, scores are not quite as high.


### Model arguments
* `--kernel str list` : type of kernels to use. 
* `--fm int` : number of node feature maps at each layer
* `--depth int` : number of layers in the network

Statistics will be saved after every epoch. Plots are updated if the network improves on its best (1/FPR) test score. If the current (1/FPR) score matches the previous best, plots are updated only if test AUC is improved upon.

## NYU Data

##### Training data
This data can be loaded with pickle, using:\
`X, y = pickle.load(open("antikt-kt-train-gcnn.pickle", "rb"), encoding="latin1")`

where
- X is a list of numpy arrays, each of which represents the (final)
constituents of the jet. That is, X[i] is a (n_constituents x 9)
array, where the 9 columns respectively correspond to p, eta, phi, E,
pt, theta, px, py, pz.
- y is the list of labels (0 or 1s)

##### Cropped and weighted training and testing data for testing
Use `python script/projectNYU/prepare_data_nyu.py` to create pickle files cropped and weighted from corresponding hdf5 files. 

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
    |   |-- clusEta  # https://en.m.wikipedia.org/wiki/Pseudorapidity
    |   |-- clusPhi  # https://en.m.wikipedia.org/wiki/Azimuth
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

