import pickle

def load_raw_data(filepath, nb_ex):
  """Loads data from the IceCube project"""
  with open(filepath, 'rb') as filein:
    X, y, weights = pickle.load(filein)
  x_t = []
  w = []
  for i in range(min(nb_ex,len(X))):
    x_t.append(X[i].transpose())
    w.append(float(weights[i]))
  return x_t[:nb_ex], y[:nb_ex], w[:nb_ex]
