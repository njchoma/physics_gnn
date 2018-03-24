import logging
from os.path import join
import matplotlib; matplotlib.use('Agg')  # no display on clusters 
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

import loading.model.model_parameters as param


def _var2tensor(tensor):
    if isinstance(tensor, Variable):
        tensor = tensor.data
    return tensor

def roc_score(gt, pred, weights=None):

    # pred = convert_bool_or_conf_to_int(pred)
    if weights is None:
        return roc_auc_score(gt, pred)
    return roc_auc_score(gt, pred, sample_weight=weights)


def roc_vals(gt, pred, weights=None):

    # pred = convert_bool_or_conf_to_int(pred)
    if weights is None:
        fpr, tpr, thresholds = roc_curve(gt, pred)
    else:
        fpr, tpr, thresholds = roc_curve(gt, pred, sample_weight=weights)

    return dict(fpr=fpr, tpr=tpr, thresholds=thresholds)


def fpr_ppercent(tprs, fprs, p=0.5):
    for i, tpr in enumerate(tprs):
        if (i + 1) % 5000 == 0:
            print('TPR nb {} : {}'.format(i + 1, tpr))
        if tpr > p:
            return fprs[i]
    raise ValueError('No TPR > {} found'.format(p))


def plot_roc_curve(gt, pred, weights, type_, save_path, model_name, zooms=None):
  """plots the ROC curve. zooms is a list of maximum range for x"""

  def _plot_roc_curve(roc, name='', xlim=[0, 1], ylim=[0, 1]):
    # plot
    plt.clf()
    plt.figure(1)
    plt.clf()
    plt.title('{} ROC Curve {}'.format(type_, name))
    plt.plot(roc["fpr"], roc["tpr"])

    # zooms
    plt.ylim(ylim)
    plt.xlim(xlim)

    # legends & labels
    plt.xlabel("False Positive Rate (1- BG rejection)")
    plt.ylabel("True Positive Rate (Signal Efficiency)")
    plt.grid(linestyle=':')

    # save
    filepath = join(save_path, '{}_roc_{}.png'.format(name, type_))
    plt.savefig(filepath)
    plt.clf()

  # signal preds
  roc = roc_vals(gt, pred, weights)
  if zooms is None:
    _plot_roc_curve(roc, name=model_name)
  else:
    for i, zoom in enumerate(zooms):
      _plot_roc_curve(roc, name=model_name + '_zoomed' + str(i), xlim=[0, zoom])


def _get_fpr(gt, pred, weights, p=0.5):
  # signal preds
  roc = roc_vals(gt, pred, weights)
  fpr = fpr_ppercent(roc['tpr'], roc['fpr'], p)
  return fpr


class ROCCurve():
  """ROC curve like statistics"""

  def __init__(self, type_, zooms=[1.0], p=0.5):
    self.zooms = zooms
    self.type_ = type_
    self.name = param.args.name
    self.savedir = param.args.savedir
    try:
      self.p = param.args.tpr_target
    except:
      self.p = p
    logging.warning("{} 1/FPR to be evaluated at {} TPR".format(self.name, self.p))

  def reset(self):
    self.gt = []
    self.pred = []
    self.weights = []

  def update(self, output, label, weights=None):
    """adds predictions to ROC curve buffer"""

    self.gt.extend([int(l) for l in label])
    self.pred.extend(output)
    if weights is not None:
      self.weights.extend(weights)

  def score_auc(self, is_weighted=True):
    """returns Area Under Curve for data in buffer"""

    if is_weighted:
      return roc_score(self.gt, self.pred, self.weights)
    return roc_score(self.gt, self.pred, None)

  def score_fpr(self):
    fpr50 = _get_fpr(self.gt, self.pred, self.weights, self.p)
    return fpr50

  def plot_roc_curve(self):
    plot_roc_curve(
                    self.gt, 
                    self.pred, 
                    self.weights, 
                    self.type_, 
                    self.savedir, 
                    self.name, 
                    zooms=self.zooms)
