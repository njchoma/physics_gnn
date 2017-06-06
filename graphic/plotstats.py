from os.path import join
import matplotlib.pyplot as plt
from csv import reader


def read_datafile(filename):
    with open(filename, 'r') as f:
        data = list(reader(f))[0]
    data = [float(value) for value in data if len(value) > 0]
    return data


def statpath(statdir, stat):

    return join(statdir, '{}.csv'.format(stat))


def plot_statistics(param, stats=None, figsize=(20, 50)):
    possible_stats = ['loss_step'] + param.possible_stats
    print(possible_stats)

    if stats is None:
        stats = possible_stats

    fig = plt.figure(figsize=figsize)
    for idx, stat in enumerate(stats):
        data = read_datafile(statpath(param.statdir, stat))

        ax = plt.subplot(len(stats), 1, idx + 1)
        ax.plot(data, label=stat)
        ax.legend(loc="upper right")
        ax.grid(linestyle=':')
        if 'loss' in stat:
            ax.set_yscale('log')

    loc = join(param.statdir, 'stats.png')
    fig.savefig(loc)
    print('saved in `{}`.'.format(loc))
    plt.show()
