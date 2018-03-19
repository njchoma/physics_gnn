import numpy as np
from random import shuffle

def get_batches(nb_samples_in, batch_size, shuffle_batch=False):
  # Note: The operator // is floor division
  nb_batches = nb_samples_in // batch_size
  last_batch_size = nb_samples_in % batch_size
  idx = list(range(nb_samples_in))
  if (shuffle_batch==True):
    shuffle(idx)
  idx_list = []
  for i in range(0,nb_batches*batch_size, batch_size):
    idx_list.append(idx[i:i+batch_size])
  idx_list.append(idx[-last_batch_size:])
  return idx_list

def pad_batch(X):
  nb_samples = len(X)
  nb_features = X[0].shape[0]
  '''
  # Do nothing if only one sample
  if (nb_samples == 1):
    return X, None
  '''

  largest_size = 0
  sample_sizes = np.zeros(nb_samples,dtype=int)
  for i, sample in enumerate(X):
    sample_sizes[i] = sample.shape[1]
    largest_size = max(largest_size, sample_sizes[i])

  pad_sizes = largest_size - sample_sizes
  mask = np.zeros(shape=(nb_samples, largest_size, largest_size))
  # Pad samples with zeros
  for i in range(nb_samples):
    zeros = np.zeros(shape=(nb_features, pad_sizes[i]))
    X[i] = np.concatenate((X[i], zeros),axis=1)
    mask[i,:sample_sizes[i],:sample_sizes[i]] = 1
  return X, mask


if __name__ == "__main__":
  n = 10
  b = 3
  batch_idx = get_batches(n,b,True)
  print(batch_idx)
