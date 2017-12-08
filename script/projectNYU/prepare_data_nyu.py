import pickle
import numpy as np
import h5py as h5

def get_weights(data, label):
    """generates weights for NYU data"""

    # Cropping
    data, label = _cropping(data, label)
    label = np.array(label)

    # Weights for flatness in pt
    weight = np.zeros(len(label))

    data0 = [data[i] for i in range(len(label)) if label[i] == 0]
    pdf, edges = np.histogram([j["pt"] for j in data0], density=True, range=[250, 300], bins=50)
    pts = [j["pt"] for j in data0]
    indices = np.searchsorted(edges, pts) - 1
    inv_w = 1. / pdf[indices]
    inv_w /= inv_w.sum()
    weight[label == 0] = inv_w

    data1 = [data[i] for i in range(len(label)) if label[i] == 1]
    pdf, edges = np.histogram([j["pt"] for j in data1], density=True, range=[250, 300], bins=50)
    pts = [j["pt"] for j in data1]
    indices = np.searchsorted(edges, pts) - 1
    inv_w = 1. / pdf[indices]
    inv_w /= inv_w.sum()
    print(inv_w.shape)
    weight[label == 1] = inv_w
    print(weight.shape)

    return data, label, weight


def _cropping(data, label):
    data_ = [j for j in data if 250 < j["pt"] < 300 and 50 < j["mass"] < 110]
    label_ = [label[i] for i, j in enumerate(data) if 250 < j["pt"] < 300 and 50 < j["mass"] < 110]
    return data_, label_


def _transfer_one_event(filein, ev_name):
    event = filein[ev_name]
    if ev_name.endswith('000'):
        print(ev_name)

    # label
    label = event.attrs['label']

    # weight related
    jet_pt = event.attrs['jet_pt']
    jet_mass = event.attrs['jet_mass']

    # features
    length = event.attrs['jet_length']
    data = np.empty([length, 6])

    data[:, 0] = event['p'][()]
    data[:, 1] = event['eta'][()]
    data[:, 2] = event['phi'][()]
    data[:, 3] = event['E'][()]
    data[:, 4] = event['pt'][()]
    data[:, 5] = event['theta'][()]

    event_descr = {'data': data, 'pt': jet_pt, 'mass': jet_mass}
    return (event_descr, label)


def _transfer_one_file(pathin):
    filein = h5.File(pathin, 'r')
    event_list = []
    event_list = [_transfer_one_event(filein, event) for event in filein]
    filein.close()
    data_list = [data for data, _ in event_list]
    label_list = [label for _, label in event_list]
    return data_list, label_list


def main():
    pathin = '/data/grochette/data_nyu/antikt-kt-test.h5'
    pathout = '/home/nc2201/research/GCNN/antikt-kt-test.pickle'

    data, label = _transfer_one_file(pathin)
    data, label, weight = get_weights(data, label)
    data = [x['data'] for x in data]

    with open(pathout, 'wb') as fileout:
        pickle.dump(output, fileout)

if __name__ == '__main__':
    main()
