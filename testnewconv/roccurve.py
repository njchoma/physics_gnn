from os.path import join
import matplotlib; matplotlib.use('Agg')  # no display on clusters 
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from torch.autograd import Variable


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

    def _plot_roc_curve(name='', xlim=[0, 1], ylim=[0, 1]):
        # signal preds
        roc = roc_vals(gt, pred, weights)
        fpr50 = fpr_ppercent(roc['tpr'], roc['fpr'], 0.5)

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
        print('saving in `{}`'.format(filepath))
        plt.savefig(filepath)
        plt.clf()

        return fpr50

    if zooms is None:
        fpr50 = _plot_roc_curve(name=model_name)
    else:
        for i, zoom in enumerate(zooms):
            fpr50 = _plot_roc_curve(name=model_name + '_zoomed' + str(i), xlim=[0, zoom])

    return fpr50


class ROCCurve():
    """ROC curve like statistics"""

    def __init__(self):
        self.gt = []
        self.pred = []
        self.weights = []

    def update(self, output, label, weights=None):
        """adds predictions to ROC curve buffer"""

        self.gt.extend([int(l) for l in _var2tensor(label)])
        self.pred.extend(_var2tensor(output))
        if weights is not None:
            self.weights.extend(_var2tensor(weights))

    def roc_score(self, is_weighted):
        """returns Area Under Curve for data in buffer"""

        ground_truth = [int(l) for l in self.gt]
        if is_weighted:
            return roc_score(ground_truth, self.pred, self.weights)
        return roc_score(ground_truth, self.pred, None)

    def plot_roc_curve(self, name, type_, save_path, zooms=None):
        ground_truth = [int(l) for l in self.gt]
        fpr50 = plot_roc_curve(ground_truth, self.pred, self.weights, type_, save_path, name, zooms=zooms)
        return fpr50