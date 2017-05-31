import utils.tensors_op as t
from utils.files import print_


def getweightfunc(param):
    """a weightfunc is a function that takes arguments and returns
    new weights. All unused arguments should be ignored"""

    choice = param.weightfunc
    if choice is None:
        return None
    elif choice == 'balance_class':
        print_('using `balance_class` weight modification with ' +
               '`avg_weight0`={} and `avg_weight1`={}'.format(
                   param.weight_average_zero, param.weight_average_one))
        return balance_class
    else:
        raise NotImplementedError('Weight functions not implemented yet.')


def balance_class(param, weight, label, *args, **kwargs):
    """returns weights such that the two class have equal cummulated weight"""

    weight_one = t.mult(weight, label) / param.weight_average_one
    weight_zero = t.mult(weight, 1 - label) / param.weight_average_zero
    weight_zero = weight_zero * param.ratio_1to0

    return weight_zero + weight_one
