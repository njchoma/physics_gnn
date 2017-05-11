import torch.nn.functional as F


def softmax(input, axis=0):
    """Apply softmax on input at certain axis.
    Parammeters:
    input: Tensor (N*L or rank>2)
    axis: the axis to apply softmax, sum over that axis sums to 1

    Returns: Tensor with softmax applied on that dimension.
    """

    input_size = input.size()

    trans_input = input.transpose(axis, len(input_size) - 1)
    trans_size = trans_input.size()

    input_2d = trans_input.contiguous().view(-1, trans_size[-1])
    soft_max_2d = F.softmax(input_2d)

    soft_max_nd = soft_max_2d.view(*trans_size)

    return soft_max_nd.transpose(axis, len(input_size) - 1)
