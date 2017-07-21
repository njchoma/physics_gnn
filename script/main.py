import os.path as path
import read_args as ra
from projectNYU.main_nyu import main_nyu
from projectNERSC.main_nersc import main_nersc


def main():
    """Reads args and chooses between main_nyu and main_nersc"""

    args = ra.read_args()
    project_root_dir = path.dirname(path.abspath(path.join(__file__, '..')))
    if args.data == 'NYU':
        main_nyu(args, project_root_dir)
    elif args.data == 'NERSC':
        main_nersc(args, project_root_dir)
    else:
        raise ValueError('--data should be NYU or NERSC')


if __name__ == '__main__':
    main()
