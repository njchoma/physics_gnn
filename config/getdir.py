from os.path import join, expanduser


def home():
    return expanduser('~')


def islocal():
    return ('gaspar' in home())


def getdatadir():
    if islocal():
        return '/home/gaspar/Data/Documents/Stage/Data/'
    else:
        return join(home(), 'data')


def getnetdir():
    if islocal():
        return '/home/gaspar/Bureau/Stage/LHC_GNN/models/'
    else:
        return join(home(), 'lhc_gnn/models')


def getstdout():
    if islocal():
        return None
    else:
        return join(home(), 'lhc_gnn/advance/')
