import numpy as np
from random import shuffle

def _sort_batch(nb_samples_in, batch_size, idx):
  nb_batches = nb_samples_in // batch_size
  idx_list = []
  for i in range(0,nb_batches*batch_size, batch_size):
    idx_list.append(idx[i:i+batch_size])
  return idx_list
  

def get_batches(nb_samples_in, batch_size, shuffle_batch=False):
  # Note: The operator // is floor division
  idx = list(range(nb_samples_in))
  if (shuffle_batch==True):
    shuffle(idx)
  return _sort_batch(nb_samples_in, batch_size, idx)


def get_sorted_batches(nb_samples_in, batch_size, X, shuffle_batch=False):
  '''
  Gets batches for testing sets, grouping together
  batches of similar size.
  This allows speedup in testing, where sample
  order doesn't matter.
  '''
  # Sort samples by nb_nodes
  sample_sizes = np.zeros(nb_samples_in)
  for i, sample in enumerate(X):
    sample_sizes[i] = sample.shape[1]
  sm_to_lg = np.argsort(sample_sizes)

  # Get batches
  idx_list = _sort_batch(nb_samples_in, batch_size, sm_to_lg)

  # Optionally shuffle order of batches
  if shuffle_batch == True:
    shuffle(idx_list)
  return idx_list


def pad_batch(X, nb_extra_nodes=0):
  nb_samples = len(X)
  nb_features = X[0].shape[0]

  largest_size = 0
  sample_sizes = np.zeros(nb_samples,dtype=int)
  for i, sample in enumerate(X):
    sample_sizes[i] = sample.shape[1]
    largest_size = max(largest_size, sample_sizes[i])

  largest_size += nb_extra_nodes
  pad_sizes = largest_size - sample_sizes 
  sample_sizes += nb_extra_nodes
  mask = np.zeros(shape=(nb_samples, largest_size, largest_size))
  # Pad samples with zeros
  for i in range(nb_samples):
    zeros = np.zeros(shape=(nb_features, pad_sizes[i]))
    X[i] = np.concatenate((X[i], zeros),axis=1)
    mask[i,:sample_sizes[i],:sample_sizes[i]] = 1
  return X, mask, sample_sizes

if __name__ == "__main__":
  n = 10
  b = 3
  batch_idx = get_batches(n,b,True)
  print(batch_idx)
