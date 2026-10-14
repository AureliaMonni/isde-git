from pandas import read_csv
import numpy as np


def load_data(filename):
    """
    Load data from a csv file

    Parameters
    ----------
    filename : string
        Filename to be loaded.

    Returns
    -------
    X : ndarray
        the data matrix.

    y : ndarray
        the labels of each sample.
    """
    data = read_csv(filename)
    z = np.array(data)
    y = z[:, 0]
    X = z[:, 1:]
    return X, y


def split_data(x, y, tr_fraction=0.5):
    n_samples = x.shape[0]
    n_tr = int(tr_fraction * n_samples)
    n_ts = n_samples - n_tr

    idx = np.arange(n_samples)
    np.random.shuffle(idx)

    tr_idx = idx[:n_tr]
    ts_idx = idx[n_tr:]

    Xts = x[ts_idx, :]
    yts = y[ts_idx]

    Xtr = x[tr_idx, :]
    ytr = y[tr_idx]

    return Xtr, Xts, ytr, yts
