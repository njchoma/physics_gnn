from os import path, listdir


def main():
    """transfers multiple files from old to new format"""
    filedir = 'models'
    filenames = listdir(filedir)
    filenames = ['.'.join(file.split('.')[:-1]) for file in filenames if file.endswith('.out')]
    for filename in filenames:
        _transfer_file(path.join(filedir, filename))


def _transfer_file(filename):
    line_acc = ['Learning Rate, Train Loss, Test Loss, Train AUC Score, Test AUC score']
    for line in open(filename + '.out', 'r'):
        if line.startswith('Loss'):  # first line, labels
            pass
        elif line:  # non empty line
            line_acc.append(_transfer_line(line))
    with open(filename + '.csv', 'w') as fileout:
        fileout.write('\n'.join(line_acc))


def _transfer_line(line):
    loss, train_auc, test_auc, lrate = [val.strip() for val in line.split(',')]
    new_vals = [lrate, loss, '=NA()', train_auc, test_auc]
    return ','.join(new_vals)


if __name__ == '__main__':
    main()
