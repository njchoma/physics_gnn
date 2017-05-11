import numpy as np
from math import sqrt
from torch.autograd import Variable


def tensor(variable):
    if isinstance(variable, Variable):
        return variable.data
    return variable


def add(t1, t2):
    return tensor(t1) + tensor(t2)


def sub(t1, t2):
    return tensor(t1) - tensor(t2)


def mult(t1, t2):
    return tensor(t1) * tensor(t2)


def div(t1, t2):
    return tensor(t1) / tensor(t2)


def mean(t, *args):
    """computes the mean along dimensions listed after `t`"""
    if any(type(t) is tp for tp in [float, int]):
        return t
    if len(args) == 0:
        return tensor(t).mean()
    args = list(args)
    args.sort(reverse=True)
    t = tensor(t)
    for dim in args:
        t = t.mean(dim)
    return t


def center(t, *args):
    t = tensor(t)
    m = mean(t, *args)
    if type(m) is not float:
        m = m.expand_as(t)
    return t - m


def var(t, *args):
    t = center(t, *args)
    t2 = t * t
    if len(args) == 0:
        return t2.mean()
    args = list(args)
    args.sort(reverse=True)
    for dim in args:
        t2 = t2.mean(dim)
    return t2


def std(t, *args):
    v = var(t, *args)
    if type(v) is float:
        return sqrt(v)
    return v.sqrt()


def normalize(t, *args):
    t = center(t, *args)
    v = var(t, *args).expand_as(t)
    if type(v) is not float:
        v = v.expand_as(t)
    return t / v


def numpy(t, squeeze=False):
    if type(t) is float:
        return np.array(t)
    if t.is_cuda:
        t = t.cpu()
    t = tensor(t)
    if squeeze:
        t = t.squeeze()
    return tensor(t).numpy()


def print_mean(t, *args):
    m = numpy(mean(t, *args), squeeze=True)
    print('mean :\n{}'.format(m))


def print_std(t, *args):
    v = numpy(std(t, *args), squeeze=True)
    print('std :\n{}'.format(v))


def print_mean_std(t, *args):
    print_mean(t, *args)
    print_std(t, *args)
