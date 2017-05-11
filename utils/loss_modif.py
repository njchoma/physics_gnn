import torch.nn as nn


class HingeEmbeddingLoss(nn.Module):
    """calls HingeEmbeddingLoss after replacing labels from {0, 1}
    to {-1, 1}"""

    def __init__(self):
        super(HingeEmbeddingLoss, self).__init__()
        self.criterion = nn.HingeEmbeddingLoss()

    def forward(self, output, label):
        return self.criterion(-output, 2 * label - 1)