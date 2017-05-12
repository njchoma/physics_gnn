# GCNN
graph-convolutions based neural networks and a few applications


### Use of _main.py_
`main.py` allows to create new models, recover already trained ones, plot training statistics or print a model's description.
Type `main.py --help` for a list of possible arguments.

##### Specify model and action
`--model` and `--mode` are the most import argument : they specify the model used and what to do with it.
`python main.py --model your_model_name --mode action`
will check for an existing `your_model_name` model in storage and use it if available, create it if not.
In all cases, `action` indicates what the program should do with the model :
- `train` for training
- `test` for testing
- `plot` for ploting the statistics aquired during training (loss, average and standard deviation for each class, ...)
- `description` for printing the model's description
- `setdefault` not implemented yet, will edit the `config/local.txt` file.

##### Cuda
Add argument `--cuda` to train or test on GPU, CPU will be used otherwise.

##### Specify architecture
If the model needs to be created, arguments can be used to specify the architecture used.
`--modeltype` specifies the type of architectures used by the new model, and add the argument `--batchnorm` if you want the network to normalize each feature map between layers. 
- `--dim` is a list of number of feature maps
- `--deg` is a list of degrees for the graph convolution poynomials

The following arguments replace the `dim` and `deg` :
- `--nb_layer` is the number of layer
- `--deg_layer` is the degree of each layer's polynomial
- `--feature_maps` is the number of feature maps for each layer

Similar arguments exist for the modification layer architecture.

##### Training parameters
You can specify an initial learning rate, the rate of learning rate decay or the time window for that decay.
You can also modify the number of epochs for training, the number of batch to be used (especially usefull for quick testing), the frequency at which statistics will be printed...


### Default organization
You can specify local paths in `config/local.txt` instead of specifying it every time you use the `main`.
If you don't, the `datadir`, `netdir` and `stdout` will take their default value, and prints won't be redirected.
Those default values rely on this code being in a directory, on the same level as :
- `data` : a directory (or link to a directory) containing the data in batch form
- `models` : a storage directory for previously trained model. Any trained model will be saved there.

```
your_project/
    |-- GCNN/
    |       |...
    |
    |-- data/
    |       |...  # explained below
    |
    |-- models/
            |...  # explained below
```

### Data structure
Use `--datadir` to specify a directory containing data organized in batch. It should be organized this way to work with the batch generator provided in `Atlas`:
```
datadir/
    |-- train/  # contains everything concerning the training set
    |       |-- weightfactors.pkl
    |       |-- len2namenum.pkl
    |       |-- data.h5  # hdf5 file containing all training batch
    |               |-- nb_events  # total number of events in data.h5
    |               |-- nb_batch   # total number of batch in data.h5
    |               |-- batch0     # one batch of data
    |               |       |-- E       # energy
    |               |       |-- EM      # electro magnetic energy
    |               |       |-- eta     # pseudorapidity
    |               |       |-- phi     # azimuth
    |               |       |-- label   # label
    |               |       |-- weight  # weight
    |               |
    |               |-- batch1
    |               |       |...
    |               |
    |               |...
    |               |
    |               |-- batch{maxbatch}
    |                       |...
    |
    |-- test/  # contains everything concerning the testing set
            |-- weightfactors.pkl
            |-- len2namenum.pkl
            |-- data.h5
                    |...
```
The batch size does not have to be consistant, but because data is represented as a Tensor, all events in one batch must be of the same size.
Regrouping events of the same size and mixing signal data and background data is the purpose of the files in `data_prepare`.


### Model storage structure
Use `--netdir` to specify a directory used to search existing models, and to store created models. Created models will be organized this way :
```
netdir/
    |-- your_model_name/
    |       |-- model            # the model itself
    |       |-- description.txt  # description
    |       |-- stat             # directory containing statistics from training
    |       |-- graphic          # directory containg ROC curves from that model
    |
    |-- other_models/
    |...
```
When using the mode `plot`, a plot of different training statistics aquired during training will be created in the `stat` directory.
