

def pad_samples(X):
  nb_samples = len(X)
  # Do nothing if only one sample
  if (nb_samples == 1):
    return X, None

  largest_size = 0
  for sample in X:
    largest_size = max(largest_size, len(sample))


