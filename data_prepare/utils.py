from os import path, mkdir, makedirs


def print_(text, stdout=None):
    """if not output file is given: prints, else: write in file"""
    if stdout is None:
        print(text)
    else:
        with open(stdout, 'a') as fout:
            fout.write(text + '\n')


def makedir_if_not_there(dirname):
    if not(path.exists(dirname)):
        try:
            mkdir(dirname)
        except OSError:
            makedirs(dirname)
