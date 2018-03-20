import time
import os
import logging
import numpy as np
from scipy.sparse import csgraph
import scipy.misc

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from utils.in_out import make_dir_if_not_there
import graphics.graph_utils as graph_utils

def construct_plot(args):
  if args.plot == None:
    return None
  logging.info("Building plotter...")
  opto_args = {}
  if args.plot == 'spectral':
    plot_class = Spectral_Plot
  elif args.plot == 'spectral3d':
    plot_class = Spectral_Plot
    opto_args['dim'] = 3
  elif args.plot == 'eig':
    plot_class = Eig_Plot
  elif args.plot == 'ker':
    plot_class = Visualize_Kernel
  else:
    raise ValueError("{} plot type not recognized".format(args.plot))
  logging.info("{} plotting to be performed".format(args.plot))
  return plot_class(args,**opto_args)


class Plot(object):
  def __init__(self, args, **kwargs):
    self.args = args
    self.nb_samples = args.nbtrain
    self.nb_layers = args.nb_layer
    self.plot_dir = os.path.join(args.savedir, 'plots')
    make_dir_if_not_there(self.plot_dir)

  def _savefig(self, name):
    fileout = os.path.join(self.plot_dir, name)
    plt.savefig(fileout, dpi=200)
    plt.clf()

  def epoch_finished(self):
    logging.warning("Plotting complete. Exiting.")
    exit()
    

class Spectral_Plot(Plot):
  def __init__(self, args, dim=2, **kwargs):
    super(Spectral_Plot, self).__init__(args)
    self.dim = dim

  def _euclidean_plot(self, nodes, edges, layer_num):
    n_pts = nodes.shape[0]
    edge_weights = edges.flatten()
    # Normalize edge weights
    edge_weights = graph_utils.normalize_edges(edge_weights)
    # Get cartesian product of edges between all node pairs
    edges = graph_utils.cartesian(np.arange(n_pts).reshape(-1,1))
    # Sort edges from lightest to darkest
    # This allows darker lines to be plotted over lighter ones
    plot_order = np.argsort(edge_weights)
    edges = edges[plot_order]
    edge_weights = edge_weights[plot_order]

    # Ensure white edges don't block gridlines
    edge_weights, edges = graph_utils.remove_white_edges(edge_weights, edges)

    # Set edge colors
    colors = [3*(1-wt,) for wt in edge_weights]

    # Perform all plotting
    if self.dim == 2:
      LineColl = LineCollection
    else:
      LineColl = Line3DCollection
    lc = LineColl(nodes[edges], colors=colors)
    fig = plt.figure()
    node_cmap = np.sum(nodes,axis=1)
    if self.dim == 2:
      ax = plt.gca()
      ax.add_collection(lc)
      ax.plot(nodes[:,0],nodes[:,1], 'ro')
    else:
      ax = plt.gca(projection='3d',zorder=1)
      ax.add_collection3d(lc)
      ax.set_xlim([nodes[:,0].min(),nodes[:,0].max()])
      ax.set_ylim([nodes[:,1].min(),nodes[:,1].max()])
      ax.set_zlim([nodes[:,2].min(),nodes[:,2].max()])
      ax.scatter3D(nodes[:,0],nodes[:,1], nodes[:,2],c=node_cmap, cmap='autumn',zorder=5)
    plt.title("Gaussian kernel, GCNN layer {}".format(layer_num))
    self._savefig("spectral_{}d_layer_{}.png".format(self.dim,layer_num))

    # Quit if last layer reached
    if layer_num==(self.nb_layers-1):
      self.epoch_finished()

  def plot_graph(self, nodes, edges, layer_num):
    edges = graph_utils.gaussian_kernel(nodes, N=2*10**1)
    edges_sum = edges.sum(axis=1)
    vk = Visualize_Kernel(self.args)
    vk.plot_graph(nodes, edges, layer_num-1)
    logging.info("Plotting layer {}".format(layer_num))
    if (edges != edges.transpose()).any():
      logging.warning("Plotting asymetric matrix. Symmetrizing.")
      edges = 0.5*edges+0.5*edges.transpose()

    # L = graph_utils.get_normed_laplacian(edges)
    L = csgraph.laplacian(edges,normed=True)
    w,v = np.linalg.eigh(L)

    # Add first num_eigvecs to spectral nodes
    # Accounts for 2d and 3d cases
    spec_nodes = v[:,1:2]
    for i in range(2,self.dim+1):
      spec_nodes = np.concatenate((spec_nodes,v[:,i:i+1]),1)
    self._euclidean_plot(spec_nodes,edges,layer_num)


class Eig_Plot(Plot):
  def __init__(self, args, nb_eigvals=20, **kwargs):
    super(Eig_Plot, self).__init__(args)
    self.nb_eigvals = nb_eigvals
    self.all_eigvals = np.zeros(shape=(args.nb_layer, self.nb_eigvals))
    self.trace = 1.0

  def plot_graph(self, nodes, edges, layer_num):
    edges = graph_utils.gaussian_kernel(nodes, N=2*10**1)
    eigvals = np.linalg.svd(edges, compute_uv=False)
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
    plt.title("Layer {} Eigenvalues".format(layer))
    plt.xticks(y_pos,np.arange(1,nb_eig+1))
    plt.xlim([-1,nb_eig])
    plt.ylim(ymin=0.0, ymax=1.0)
    self._savefig("eig_layer_{}.png".format(layer))

  def epoch_finished(self):
    self.all_eigvals /= np.max(self.all_eigvals)
    for i in range(self.nb_layers):
      logging.info("Plotting eigenvalues, layer {}".format(i))
      self._bar_plot(self.all_eigvals[i], i)
    logging.warning("All plotting complete. Exiting")
    exit()

class Visualize_Kernel(Plot):
  def __init__(self, args, **kwargs):
    super(Visualize_Kernel, self).__init__(args)

  def _print_kernel_stats(self, edges, nb_print=6):
    def _log_info(size, size_type,  values):
      logging.info("{} {} edge weight {}:".format(nb_print, size, size_type))
      logging.info(values)
    edge_sums = edges.sum(1)
    sorted_idx = np.argsort(edge_sums)
    _log_info("smallest", "row sums", edge_sums[sorted_idx[:nb_print]])
    _log_info("smallest", "positions", sorted_idx[:nb_print])
    _log_info("largest", "row sums", edge_sums[sorted_idx[-nb_print:]])
    _log_info("largest", "positions", sorted_idx[-nb_print:])

  def plot_graph(self, nodes, edges, layer_num):
    name = "kernel_layer_{}.jpg".format(layer_num)
    #fileout = os.path.join(self.plot_dir, name)
    plt.imshow(edges)
    #scipy.misc.imsave(fileout, edges)
    self._print_kernel_stats(edges)
    self._savefig(name)
    if layer_num==(self.nb_layers-1):
      self.epoch_finished()
