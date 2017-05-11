import torch


def batchnorm(x, axis=0):
    """renormalize independantly each column of input.
    input : x variable of size batch * 1 * n * d_out
    output : y = (x - E(x)) / sqrt(var(x)) of same size
    """

    # Compute empirical expectancy and substract
    ex = x.mean(axis)  # axis representing the event length
    x = x - ex.expand_as(x)

    # Compute empirical standard deviation and divide
    std = torch.sqrt((x * x).mean(axis))
    std = std + std.mean().expand_as(std) / 10000  # avoids division by zero
    x = x / std.expand_as(x)

    return x
