from os.path import join, exists
from os import mkdir, makedirs, listdir
import h5py as h5
from shutil import copyfile
import multiprocessing


def makedir_if_not_there(dirname):
    if not exists(dirname):
        try:
            mkdir(dirname)
        except OSError:
            makedirs(dirname)


def makefile_if_not_there(dirname, filename, text=None):
    makedir_if_not_there(dirname)
    filepath = join(dirname, filename)
    if not exists(filepath):
        with open(filepath, 'w') as fout:
            if text is not None:
                fout.write(text)


def transfer_batch(eventname, filein, fileout):
    important_fields = ['clusE', 'clusEM', 'clusPhi', 'clusEta', 'weight']
    if eventname.startswith('event'):
        group = fileout.create_group(eventname)
        for field in important_fields:
            group.create_dataset(field, data=filein[eventname][field])


class Transfer_file:
    """selects and copys `clusE`, `clusEM`, `clusPhi`, `clusEta`
    and `weight` from `rawdatadir/filename` and stores the result
    in `datadir`
    """

    def __init__(self, rawdatadir, datadir):
        self.rawdatadir = rawdatadir
        self.datadir = datadir
        self.transfered_path = join(self.datadir, 'transfered.txt')

        with open(self.transfered_path, 'r') as transfered_file:
            self.transfered = transfered_file.read().split('\n')

    def __call__(self, filename):
        if filename not in self.transfered:
            self.transfer(filename)
            print('{: >40}: DONE'.format(filename))
        else:
            print('{: >40}: DONE PREVIOUSLY'.format(filename))

    def transfer(self, filename):
        source_path = join(self.rawdatadir, filename)
        target_path = join(self.datadir, filename)

        if not(filename.endswith('.h5')):
            copyfile(source_path, target_path)

        else:
            with h5.File(source_path, 'r') as filein:
                with h5.File(target_path, 'w') as fileout:
                    for eventname in filein:
                        transfer_batch(eventname, filein, fileout)

        with open(self.transfered_path, 'a') as transfered_file:
            transfered_file.write(filename + '\n')


def transfer(rawdatadir, datadir):
    h5_files = listdir(rawdatadir)

    pool = multiprocessing.Pool(64)
    pool.map(Transfer_file(rawdatadir, datadir), h5_files)
    print('\n' + '-' * 20 + '\nFINISHED MULTIPROCESSING\n' + '-' * 20)


if __name__ == '__main__':
    rawdatadir = '/global/homes/r/rochette/rawdata'
    datadir = '/global/homes/r/rochette/lhc_gnn/data/rawdata'
    makedir_if_not_there(datadir)
    makefile_if_not_there(datadir, 'transfered.txt')

    transfer(rawdatadir, datadir)
