import torch.nn as nn
from torch.nn.functional import relu


class MarginRankingLoss(nn.Module):
    """Creates a criterion that measures the loss given inputs
    x1, x2, two 1D mini-batch Tensors, a label 1D mini-batch
    tensor y with values (1 or 0) and a weight 1D mini-batch
    tensor w with positive values.

    If y==1 then it assumed the first input should be ranked
    higher (have a larger value) than the second input, and
    vice-versa for y==0.

    The loss function for each sample in the mini-batch is
    ```
    loss(x, y) = w * max(0, -(2 * y - 1) * (x1 - x2) + margin)
    ```

    if the internal variable `size_average = True`, the loss
    function averages the loss over the batch samples; if
    `size_average = False`, the the loss function sums over the
    batch samples. By default, `size_average` equals to `True`.
    """

    def __init__(self, margin=0, size_average=True):
        super(MarginRankingLoss, self).__init__()
        self.margin = margin
        self.size_average = size_average

    def forward(self, x, label, w):
        label = (2 * label - 1).type_as(x)  # from {0, 1} to {-1, 1}
        x = x[:, 0] - x[:, 1]
        x = - (label * x - self.margin)
        x = w * relu(x)
        return x.mean()
