import os


def print_(print_arg, quiet):
    """prints if not quiet"""

    if not quiet:
        print(print_arg)

def make_dir_if_not_there(path_to_dir):
    """Check if directory exists, creates it if not"""

    if not os.path.exists(path_to_dir):
        os.makedirs(path_to_dir)
