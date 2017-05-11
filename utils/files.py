from os import mkdir, makedirs
from os.path import join, exists


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


def print_(text, stdout=None):
    """if not output file is given: prints, else: write in file"""
    if stdout is None:
        print(text)
    else:
        with open(stdout, 'a') as fout:
            fout.write(text + '\n')
