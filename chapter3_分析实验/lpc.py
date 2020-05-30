import numpy as np


def lpc_coeff(s, p):
    """
    :param s: 一帧数据
    :param p: 线性预测的阶数
    :return:
    """
    n = len(s)
    # 计算自相关函数
    Rp = np.zeros(p)
    for i in range(p):
        Rp[i] = np.sum(np.multiply(s[i + 1:n], s[:n - i - 1]))
    Rp0 = np.matmul(s, s.T)
    Ep = np.zeros((p, 1))
    k = np.zeros((p, 1))
    a = np.zeros((p, p))
    # 处理i=0的情况
    Ep0 = Rp0
    k[0] = Rp[0] / Rp0
    a[0, 0] = k[0]
    Ep[0] = (1 - k[0] * k[0]) * Ep0
    # i=1开始，递归计算
    if p > 1:
        for i in range(1, p):
            k[i] = (Rp[i] - np.sum(np.multiply(a[:i, i - 1], Rp[i - 1::-1]))) / Ep[i - 1]
            a[i, i] = k[i]
            Ep[i] = (1 - k[i] * k[i]) * Ep[i - 1]
            for j in range(i - 1, -1, -1):
                a[j, i] = a[j, i - 1] - k[i] * a[i - j - 1, i - 1]
    ar = np.zeros(p + 1)
    ar[0] = 1
    ar[1:] = -a[:, p - 1]
    G = np.sqrt(Ep[p - 1])
    return ar, G


def lpcff(ar, npp=None):
    """
    :param ar: 线性预测系数
    :param npp: FFT阶数
    :return:
    """
    p1 = ar.shape[0]
    if npp is None:
        npp = p1 - 1
    ff = 1 / np.fft.fft(ar, 2 * npp + 2)
    return ff[:len(ff) // 2]


def lpc_lpccm(ar, n_lpc, n_lpcc):
    lpcc = np.zeros(n_lpcc)
    lpcc[0] = ar[0]  # 计算n=1的lpcc
    for n in range(1, n_lpc):  # 计算n=2,,p的lpcc
        lpcc[n] = ar[n]
        for l in range(n - 1):
            lpcc[n] += ar[l] * lpcc[n - 1] * (n - l) / n
    for n in range(n_lpc, n_lpcc):  # 计算n>p的lpcc
        lpcc[n] = 0
        for l in range(n_lpc):
            lpcc[n] += ar[l] * lpcc[n - 1] * (n - l) / n
    return -lpcc
