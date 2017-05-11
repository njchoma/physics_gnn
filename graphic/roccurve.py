from os.path import join
from utils.files import makedir_if_not_there
import matplotlib; matplotlib.use('Agg')  # no display on clusters 
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from torch.autograd import Variable


def roc_vals(gt, pred, weights=None):

    # pred = convert_bool_or_conf_to_int(pred)
    if weights is None:
        fpr, tpr, thresholds = roc_curve(gt, pred)
    else:
        fpr, tpr, thresholds = roc_curve(gt, pred, sample_weight=weights)

    return dict(fpr=fpr, tpr=tpr, thresholds=thresholds)


def plot_roc_curve(gt, pred, weights, type_, save_path, zooms=None):
    """plots the ROC curve. zooms is a list of maximum range for x"""

    def _plot_roc_curve(name='', xlim=[0, 1], ylim=[0, 1]):
        # signal preds
        roc = roc_vals(gt, pred, weights)

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
        makedir_if_not_there(save_path)
        filepath = join(save_path, '{}_roc_curve_{}.png'.format(type_, name))
        print('saving in `{}`'.format(filepath))
        plt.savefig(filepath)
        plt.clf()

    if zooms is None:
        _plot_roc_curve()
    else:
        for i, zoom in enumerate(zooms):
            _plot_roc_curve(name='zoomed' + str(i), xlim=[0, zoom])


class ROCCurve():
    """ROC curve like statistics"""

    def __init__(self):
        self.gt = []
        self.pred = []
        self.weights = []

    def _extend(self, key, arg):
        if isinstance(arg, Variable):
            arg = arg.data
        self.__dict__[key].extend(arg)

    def update(self, output, label, weights):
        self._extend('gt', label)
        self._extend('pred', output)
        self._extend('weights', weights)

    def plot_roc_curve(self, type_, save_path, zooms=None):
        ground_truth = [int(l) for l in self.gt]
        plot_roc_curve(ground_truth, self.pred, self.weights, type_, save_path, zooms=zooms)
