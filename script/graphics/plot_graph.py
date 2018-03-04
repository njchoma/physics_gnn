import numpy as np
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

def plot_nodes(nodes):
  n_pts = nodes.shape[0]
  plt.plot(nodes[:,0],nodes[:,1],'ro ')
  # plt.show()

def euclidean_plot(nodes, edges,layer_num):
  n_pts = nodes.shape[0]
  norm_edges = (edges-edges.min())/(edges.max()-edges.min())
  edge_colors = np.zeros(shape=n_pts*n_pts)
  node_pairs  = np.zeros(shape=(n_pts*n_pts,2,2))
  for i in range(n_pts):
    for j in range(n_pts):
      edge_colors[i*n_pts+j] = min(norm_edges[i,j],1.0)
      node_pairs[ i*n_pts+j] = nodes[[i,j]]
  # sort edges from lightest to darkest
  plot_order = np.argsort(edge_colors)
  for i in plot_order:
    plt.plot(node_pairs[i, :,0],node_pairs[i, :,1],color=3*(1-edge_colors[i],))
  plt.plot(nodes[:,0],nodes[:,1],'ro ')
  plt.savefig("graph_{}.png".format(layer_num))
  # plt.show()
  plt.clf()

def spectral_plot_graph(nodes, edges,layer_num):
  n_pts = nodes.shape[0]
  n_dim = len(nodes.shape)

  D = np.sum(edges,0)
  L = np.diag(D) - edges
  w,v = np.linalg.eigh(L)

  fst_eigvec = 1
  sec_eigvec = fst_eigvec+1
  spec_nodes = np.concatenate((v[:,fst_eigvec:fst_eigvec+1],v[:,sec_eigvec:sec_eigvec+1]),1)
  euclidean_plot(spec_nodes,edges,layer_num)
