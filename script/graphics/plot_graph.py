import os
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

from utils.in_out import make_dir_if_not_there

def construct_plot(args):
  logging.info("Building plotter...")
  if args.plot == 'spectral':
    plot_class = Spectral_Plot
  elif args.plot == None:
    logging.info("No plotting selected")
    return None
  else:
    raise ValueError("{} plot type not recognized".format(args.plot))
  logging.info("{} plotting to be performed".format(args.plot))
  plots = []
  for i in range(args.nb_layer):
    plots.append(plot_class(args))
  return plots

class Plot(object):
  def __init__(self, args, **kwargs):
    self.ep_finished = False
    self.nb_layers = args.nb_layer
    self.plot_dir = os.path.join(args.savedir, 'plots')
    make_dir_if_not_there(self.plot_dir)

  def _savefig(self, name):
    fileout = os.path.join(self.plot_dir, name)
    plt.savefig(fileout)
    plt.clf()

  def epoch_finished(self, layer):
    self.ep_finished = True
    if layer == self.nb_layers-1:
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
    self._savefig("spectral_{}.png".format(layer_num))
    self.epoch_finished(layer_num)

  def plot_graph(self, nodes, edges, layer_num):
    logging.info("Plotting layer {}".format(layer_num))
    n_pts = nodes.shape[0]
    n_dim = len(nodes.shape)

    D = np.sum(edges,0)
    L = np.diag(D) - edges
    w,v = np.linalg.eigh(L)

    fst_eigvec = 0
    sec_eigvec = fst_eigvec+1
    spec_nodes = np.concatenate((v[:,fst_eigvec:fst_eigvec+1],v[:,sec_eigvec:sec_eigvec+1]),1)
    self._euclidean_plot(spec_nodes,edges,layer_num)
