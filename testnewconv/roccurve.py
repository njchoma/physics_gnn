from os.path import join
import matplotlib; matplotlib.use('Agg')  # no display on clusters 
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
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
