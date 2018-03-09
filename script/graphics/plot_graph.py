import os
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from scipy.sparse import csgraph
import scipy.misc

from utils.in_out import make_dir_if_not_there

def construct_plot(args):
  logging.info("Building plotter...")
  if args.plot == 'spectral':
    plot_class = Spectral_Plot
  elif args.plot == 'eig':
    plot_class = Eig_Plot
  elif args.plot == 'ker':
    plot_class = Visualize_Kernel
  elif args.plot == None:
    logging.info("No plotting selected")
    return None
  else:
    raise ValueError("{} plot type not recognized".format(args.plot))
  logging.info("{} plotting to be performed".format(args.plot))
  return plot_class(args)

class Plot(object):
  def __init__(self, args, **kwargs):
    self.nb_samples = args.nbtrain
    self.nb_layers = args.nb_layer
    self.plot_dir = os.path.join(args.savedir, 'plots')
    make_dir_if_not_there(self.plot_dir)

  def _savefig(self, name):
    fileout = os.path.join(self.plot_dir, name)
    plt.savefig(fileout)
    plt.clf()

  def epoch_finished(self):
    logging.warning("Plotting complete. Exiting.")
    exit()
    
class Spectral_Plot(Plot):
  def __init__(self, args, **kwargs):
    super(Spectral_Plot, self).__init__(args)

  def _plot_nodes(self, nodes):
    n_pts = nodes.shape[0]
    plt.plot(nodes[:,0],nodes[:,1],'ro ')

  def _euclidean_plot(self, nodes, edges,layer_num):
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
    plt.title("Spectral embedding, layer {}".format(layer_num))
    self._savefig("spectral_{}.png".format(layer_num))
    if layer_num==(self.nb_layers-1):
      self.epoch_finished()

  def plot_graph(self, nodes, edges, layer_num):
    logging.info("Plotting layer {}".format(layer_num))
    n_pts = nodes.shape[0]
    n_dim = len(nodes.shape)

    L = edges-np.diag(edges)
    L = csgraph.laplacian(L,normed=False)
    w,v = np.linalg.eigh(L)

    fst_eigvec = 1
    sec_eigvec = fst_eigvec+1
    spec_nodes = np.concatenate((v[:,fst_eigvec:fst_eigvec+1],v[:,sec_eigvec:sec_eigvec+1]),1)
    self._euclidean_plot(spec_nodes,edges,layer_num)

class Eig_Plot(Plot):
  def __init__(self, args, nb_eigvals=10, **kwargs):
    super(Eig_Plot, self).__init__(args)
    self.nb_eigvals = nb_eigvals
    self.all_eigvals = np.zeros(shape=(args.nb_layer, self.nb_eigvals))
    self.trace = 1.0

  def plot_graph(self, nodes, edges, layer_num):
    L = edges-np.diag(edges)
    L = csgraph.laplacian(L,normed=False)
    eigvals,_ = np.linalg.eigh(L)
    # Only use first nb_eigvals
    eigvals = eigvals[:self.nb_eigvals]
    # Set trace normalization for layer 0, 1
    # Layer zero is a different kernel from other layers
    # Must perform normalization for layers 1-nb_layer too
    if layer_num in {0,1}:
      self.trace = np.sum(eigvals)
    eigvals /= self.trace
    # Update eigenvalues
    # Accounting for samples which have fewer than nb_eigvals points
    self.all_eigvals[layer_num, :eigvals.shape[0]] += eigvals

  def _bar_plot(self, eigvals, layer):
    nb_eig = eigvals.shape[0]
    y_pos = np.arange(nb_eig)
    plt.bar(y_pos, eigvals, align='center', alpha=0.5)
    plt.xlabel("Eigenvalues")
    plt.xticks(y_pos,np.arange(1,nb_eig+1))
    plt.xlim([-1,nb_eig])
    self._savefig("eig_layer_{}.png".format(layer))

  def epoch_finished(self):
    for i in range(self.nb_layers):
      logging.info("Plotting eigenvalues, layer {}".format(i))
      self._bar_plot(self.all_eigvals[i], i)
    logging.warning("All plotting complete. Exiting")
    exit()

class Visualize_Kernel(Plot):
  def __init__(self, args, **kwargs):
    super(Visualize_Kernel, self).__init__(args)

  def plot_graph(self, nodes, edges, layer_num):
    name = "kernel_layer_{}.jpg".format(layer_num)
    #fileout = os.path.join(self.plot_dir, name)
    plt.imshow(edges)
    #scipy.misc.imsave(fileout, edges)
    self._savefig(name)
    if layer_num==(self.nb_layers-1):
      self.epoch_finished()
