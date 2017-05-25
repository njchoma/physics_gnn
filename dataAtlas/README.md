##### This directory contains a subset of data to get stated with the repository.

For the repository to use this data subset, this repository should be moved to be at the same level as GCNN and renamed `data`:
```
your_project/
    |-- GCNN/
    |       |...
    |
    |-- data/
    |       |...
    |
    |-- models/
            |...
```
An other option is to provide the path to this directory in the local configuration `config/local.txt` file, as the local default for `datadir`

##### This directory contains :
events of 4 different lengths : 225, 226, 227 and 228
for a total of 3298 batch of training data (approx. 36000 events) and 371 batch of testing data (approx. 7500 events)
