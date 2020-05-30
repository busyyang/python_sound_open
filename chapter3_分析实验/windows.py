import numpy as np


def reg_window(N):
    return np.ones(N)


def hanning_window(N):
    nn = [i for i in range(N)]
    return 0.5 * (1 - np.cos(np.multiply(nn, 2 * np.pi) / (N - 1)))


def hamming_window(N):
    nn = [i for i in range(N)]
    return 0.54 - 0.46 * np.cos(np.multiply(nn, 2 * np.pi) / (N - 1))
