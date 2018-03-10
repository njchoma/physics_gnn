import logging
import numpy as np

def get_degree(A):
  D = np.sum(A,axis=1)
  D = np.diag(D)
  return D

def get_laplacian(A):
  D = get_degree(A)
  return D-A

def get_normed_laplacian(A):
  D = get_degree(A)
  I = np.identity(A.shape[0])
  D_12_inv = np.diag(np.power(np.diag(D), -0.5))
  L = I-np.matmul(D_12_inv, np.matmul(A, D_12_inv))
  return L

def gaussian_kernel(nodes, k = 0.00001, N = 15):
  # Compute pairwise distances
  coords = np.expand_dims(nodes,1)
  coords = np.repeat(coords, nodes.shape[0], 1)
  distances = coords-coords.transpose(1,0,2)
  distances = np.square(distances)
  distances = np.sum(distances,axis=2)

  # Apply gaussian kernel
  sigma_sq = np.sum(distances)*k*N
  adj = distances / sigma_sq
  adj = np.exp(-adj)
  return adj

def normalize_edges(edges):
  if edges.min() < 0.0:
    logging.warning("Edge weights below zero. Shifting for plotting")
    edges = edges-edges.min()
  if edges.max() > 1.0:
    logging.warning("Rescaling edge weights for plotting")
    edge_weights = edges/edges.max()
  if edges.max() < 0.1:
    logging.warning("Edge weights small. Rescaling for visibility")
    edges *= 0.8/edges.max()
  return edges

def cartesian(samples):
  n_repeats, dim2 = samples.shape
  a = np.tile(samples,(1,n_repeats))
  a = a.reshape(n_repeats*n_repeats,dim2)
  b = np.tile(samples,(n_repeats,1))
  return np.concatenate((a,b),1)

def remove_white_edges(edge_weights, edges, threshold=0.1):
  dark_edges = np.where(edge_weights > threshold)
  return edge_weights[dark_edges], edges[dark_edges]
