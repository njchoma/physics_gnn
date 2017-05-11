import os


def weightfactor(nbevent, nbfile, nbfileused):
    div = nbevent * nbfileused
    if div == 0:
        return 0
    factor = float(nbfile) / div
    return factor


def findtype(filename, filetypes):
    for filetype in filetypes:
        if filetype in filename:
            return filetype
    raise KeyError(
        'Type of file `{}` not found in {}'.format(filename, filetypes))


def init_weight_factors(is_used, rawdatadir):
    """reads from file `DelphesNevents` the number of such events
    and computes the renormalizing factors"""

    with open(os.path.join(rawdatadir, 'DelphesNevents'), 'r') as delphesnevents:
        eventtype = delphesnevents.read().split('\n')
    eventtype = [event for event in eventtype if len(event.strip()) > 0]  # remove empty lines

    # total number of events in data
    type2nbevent = dict()  # mapping from event type to number of events in data
    for event in eventtype:
        event = event.split(' ')  # (eventtype, nb_such_events)
        event[0] = event[0].split('_')[1]
        type2nbevent[event[0]] = int(event[1])

    filetypes = type2nbevent.keys()

    # number of files of each type
    type2nbfile = dict()
    h5_files = [filename for filename in os.listdir(rawdatadir)
                if filename.endswith('.h5')]
    for filetype in filetypes:
        type2nbfile[filetype] = sum(filetype in filename for filename in h5_files)

    # number of used files of each type
    type2nbfileused = dict()
    h5_files_used = [filename for filename in os.listdir(rawdatadir)
                     if filename.endswith('.h5') and is_used(filename)]
    for filetype in filetypes:
        type2nbfileused[filetype] = sum(filetype in filename for filename in h5_files_used)

    # weight renormalization
    type2weightfactor = {filetype: weightfactor(
        type2nbevent[filetype], type2nbfile[filetype], type2nbfileused[filetype]
    ) for filetype in filetypes}

    # filename to weight factors
    file2weightfactor = {
        filename: type2weightfactor[findtype(filename, filetypes)]
        for filename in h5_files_used
    }

    return file2weightfactor
