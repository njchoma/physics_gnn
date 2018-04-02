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

To train a network, launch `main.sh` with arguments defined as desired. Run `python3 script/main.py --help` to see a list of arguments.

#### Experiment arguments
* `--name str` : network reference name
* `--data {NYU, ICECUBE, NERSC}` : dataset to use
* `--cuda` : flag to train on GPU
* `--nbtrain int` : number of training samples to use
* `--nbtest int` : number of testing samples to use
* `--nb_batch int` : minibatch size
* `--nbprint int` : print frequency by batch (note that at this point changing minibatch size changes print frequency)

#### Model arguments
* `--fm int` : number of node feature maps at each layer
* `--depth int` : number of layers in the network
* `--lr float` : initial learning rate
* `--lrdecay [0,1]` : exponential decay factor
* `--nb_extra_nodes int` : number of zero-initialized nodes to add to experiment (see below)
* `--readout {DTNN_, ''}+{Sum, Mean, Max}` : type of operation after all graph convolution layers for transforming points into prediction
* `--node_type {Identity, GRU}` : method for updating points
* `--conv_type {Simple, ResGNN}` : graph convolution method

#### Kernel-specific arguments
* `--kernels str list` : type of kernels to use (see below)
* `--combine_kernels str` : method for combining multiple kernels together at each layer (see below)
* `--nb_MLPadj_hidden int` : only for use with MLP kernels. Number of hidden units to use

#### Optional arguments
* `--save_best_model` : flag to save best model based on test 1/FPR
* `--sorted_training` : flag to group similar-sized training samples (test does this by default). Minibatches of different sizes are padded with zeros so setting this flag significantly speeds up training. However, scores are not quite as high
* `--quiet` : flag to reduce printing
* `--no_shuffle` : flag to load and run samples in the same order. Good for plotting
* `--plot {spectral, spectral3d, eig, ker}` : type of plotting to perform
* `--tpr_target [0,1]` : set the TPR against which 1/FPR will be measures. Default is 0.5

Statistics will be saved after every epoch. Plots are updated if the network improves on its best (1/FPR) test score. If the current (1/FPR) score matches the previous best, plots are updated only if test AUC is improved upon.

## Additional Information
### Kernels
Kernels by default are computed at the first layer only and saved for use in later layers. Optional tags may be used which change the behavior of kernels. Example kernels may be `QCDAwareMeanNorm-first_only` `MLPdirected-layerwise-no_first`.
##### Kernel options
* `-layerwise` : Instantiates a kernel of specified type at every layer. Each instantiation is applied to one layer only
* `-no_first` : Kernel is instantiated at every layer except the first. Must be used with `-layerwise` tag
* `-first_only` : Kernel is used at the first layer only. May not be used with `-layerwise` tag. Must have at least one additional kernel for remaining layers

Multiple kernels may be used at each layer and combined together. Methods for combining kernels are:
##### Combine kernels
* `Fixed_Balanced` : Default option. Outputs average of all kernels
* `Affine` : Parameterized affine combination of kernels

##### Available kernels
* `QCDAwareMeanNorm` : Physics-inspired kernel which works well in practice. For use with `NYU, NERSC` data only. Not for use with `-layerwise` tag
* `GaussianSoftmax` : Computes pairwise-distance based upon spatial coordinates when used in the first layer. In later layers, uses all features
* `DistMult` : Parameterizes kernel which works well only with `Simple` convolutions and `GRU` nodes
* `MLPdirected` : Learned kernel which uses an MLP to compute pairwise distances

### Added Nodes
There is currently an anomaly (in every network architecture tested) whereby appending zero-valued points to the sample input drastically speeds up training. I am working to understand why this is occuring with the hope of removing the need for its inclusion, but at present time - on the NYU dataset - using 30 extra nodes increases 1/FPR by 40-50%.



















