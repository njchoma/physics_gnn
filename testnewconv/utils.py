from torch.autograd import Variable


def _variable_as(tensor1, tensor2):
    """Makes tensor1 a Variable depending on tensor2"""

    if isinstance(tensor2, Variable):
        tensor1 = Variable(tensor1)

    return tensor1


def _cuda_as(tensor1, tensor2):
    """Makes tensor1 cuda depending on tensor2"""

    if tensor2.is_cuda:
        tensor1 = tensor1.cuda()

    return tensor1
