import numpy as np
from scipy.signal import medfilt


def linsmoothm(x, n=3):
    win = np.hamming(n)
    win /= np.sum(win)
    l = len(x)
    y = np.zeros(l)
    if np.mod(n, 2) == 0:
        ll = n // 2
        xx = np.hstack((x[0], x, x[-1] * np.ones(ll)))
    else:
        ll = (n - 1) // 2
        xx = np.hstack((x[0], x, x[-1] * np.ones(ll + 1)))
    for i in range(l):
        y[i] = np.matmul(win, xx[i:i + n])
    return y


def pitfilterm1(x, vseg, vsl):
    y = np.zeros(len(x))
    for i in vseg.keys():
        ixb = vseg[i]['start']
        ixe = vseg[i]['end']
        u = x[ixb:ixe]
        u = medfilt(u, 5)
        v0 = linsmoothm(u, 5)
        y[ixb:ixe] = v0
    return y
