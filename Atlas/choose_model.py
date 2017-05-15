from GraphConv.architectures.complex import RGCs_FCL
from GraphConv.architectures.dummy import Dummy, DummyGconv
from GraphConv.architectures.netarch import GNN, GNNSpatial
from GraphConv.architectures.modifnetarch import GNNModif, GNNModifSpatial
from GraphConv.architectures.resnetarch import GNNRes, GNNResSpatial


def init_model_type(param):
    modeltype = param.modeltype
    if modeltype == 'Dummy':
        return Dummy(logistic_bias=param.logistic_bias), ['logistic_bias']

    # GNNs

    elif modeltype == 'DummyGconv':
        return DummyGconv(logistic_bias=param.logistic_bias), ['logistic_bias']

    elif modeltype == 'GNN':
        return GNN(param.dim, param.deg, logistic_bias=param.logistic_bias), \
            ['dim', 'deg', 'logistic_bias']

    elif modeltype == 'GNNSpatial':
        return GNNSpatial(
            param.dim, param.deg, logistic_bias=param.logistic_bias,
            normalize=param.normalize
        ), ['dim', 'deg', 'logistic_bias']

    elif modeltype == 'GNNModif':
        return GNNModif(param.dim, param.deg, param.modifdeg, logistic_bias=param.logistic_bias), \
            ['dim', 'deg', 'modifdeg', 'logistic_bias']

    elif modeltype == 'GNNModifSpatial':
        return GNNModifSpatial(
            param.dim, param.deg, param.modifdeg, logistic_bias=param.logistic_bias,
            usebatchnorm=param.batchnorm, normalize=param.normalize
        ), ['dim', 'deg', 'modifdeg', 'logistic_bias']

    # RESIDUAL GNNs

    elif modeltype == 'GNNRes':
        return GNNRes(param.dim, param.deg, usebatchnorm=param.batchnorm,
                      logistic_bias=param.logistic_bias), \
            ['dim', 'deg', 'batchnorm', 'logistic_bias']

    elif modeltype == 'GNNResSpatial':
        return GNNResSpatial(
            param.dim, param.deg, usebatchnorm=param.batchnorm,
            logistic_bias=param.logistic_bias, normalize=param.normalize
        ), ['dim', 'deg', 'batchnorm', 'logistic_bias']

    # MORE COMPLEX MODELS

    elif modeltype == 'RGCs_FCL':
        return RGCs_FCL(
            param.dim, param.deg, usebatchnorm=param.batchnorm,
            logistic_bias=param.logistic_bias, normalize=param.normalize,
            knn=param.knn
        ), ['dim', 'deg', 'batchnorm', 'logistic_bias', 'knn']

    # UNKNOWN ARGUMENT

    else:
        raise Exception('Model type not recognized : {}'.format(modeltype))
