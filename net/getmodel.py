import pickle
from os.path import join, exists
from utils.files import makedir_if_not_there
from net.model import Model


# EDIT : change pikle to torch.save

def get_model(param, stdout=None):
    """Retrieves existing model. For training, missing model will
    be created. Change model cuda mode to `cuda`"""

    # Retrieve or create model
    modelpath = join(param.netdir, 'model')

    if exists(modelpath):
        with open(modelpath, 'rb') as filein:
            model = pickle.Unpickler(filein).load()
        retrieved = True
        model.newparameters(param, stdout=stdout)
    else:
        if param.mode != 'train':  # file shouldn't be created
            raise OSError('invalid path : {}'.format(modelpath))

        # Initiate model directory
        makedir_if_not_there(param.netdir)
        model = Model(param, stdout)
        retrieved = False

    # modify model cuda mode
    if param.cuda:
        if not model.is_cuda:
            model.cuda()
    else:
        if model.is_cuda:
            model.cpu()

    return (model, retrieved)
