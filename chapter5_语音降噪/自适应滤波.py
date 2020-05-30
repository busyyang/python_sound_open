import numpy as np
from chapter3_分析实验.C3_1_y_1 import enframe


def SNR_Calc(s, r):
    """
    计算信号的信噪比
    :param s: 信号
    :param r1: 噪声
    :return:
    """
    Ps = np.sum(np.power(s - np.mean(s), 2))
    Pr = np.sum(np.power(r - np.mean(r), 2))
    return 10 * np.log10(Ps / Pr)


def LMS(xn, dn, M, mu, itr):
    """
    使用LMS自适应滤波
    :param xn:输入的信号序列
    :param dn:所期望的响应序列
    :param M:滤波器的阶数
    :param mu:收敛因子(步长)
    :param itr:迭代次数
    :return:
    """
    en = np.zeros(itr)  # 误差序列,en(k)表示第k次迭代时预期输出与实际输入的误差
    W = np.zeros((M, itr))  # 每一行代表一个加权参量,每一列代表-次迭代,初始为0
    # 迭代计算
    for k in range(M, itr):
        x = xn[k:k - M:-1]
        y = np.matmul(W[:, k - 1], x)
        en[k] = dn[k] - y
        W[:, k] = W[:, k - 1] + 2 * mu * en[k] * x
    # 求最优输出序列
    yn = np.inf * np.ones(len(xn))
    for k in range(M, len(xn)):
        x = xn[k:k - M:-1]
        yn[k] = np.matmul(W[:, -1], x)
    return yn, W, en


def NLMS(xn, dn, M, mu, itr):
    """
    使用Normal LMS自适应滤波
    :param xn:输入的信号序列
    :param dn:所期望的响应序列
    :param M:滤波器的阶数
    :param mu:收敛因子(步长)
    :param itr:迭代次数
    :return:
    """
    en = np.zeros(itr)  # 误差序列,en(k)表示第k次迭代时预期输出与实际输入的误差
    W = np.zeros((M, itr))  # 每一行代表一个加权参量,每一列代表-次迭代,初始为0
    # 迭代计算
    for k in range(M, itr):
        x = xn[k:k - M:-1]
        y = np.matmul(W[:, k - 1], x)
        en[k] = dn[k] - y
        W[:, k] = W[:, k - 1] + 2 * mu * en[k] * x / (np.sum(np.multiply(x, x)) + 1e-10)
    # 求最优输出序列
    yn = np.inf * np.ones(len(xn))
    for k in range(M, len(xn)):
        x = xn[k:k - M:-1]
        yn[k] = np.matmul(W[:, -1], x)
    return yn, W, en


def SpectralSub(signal, wlen, inc, NIS, a, b):
    """
    谱减法滤波
    :param signal:
    :param wlen:
    :param inc:
    :param NIS:
    :param a:
    :param b:
    :return:
    """
    wnd = np.hamming(wlen)
    y = enframe(signal, wnd, inc)
    fn, flen = y.shape
    y_a = np.abs(np.fft.fft(y, axis=1))
    y_a2 = np.power(y_a, 2)
    y_angle = np.angle(np.fft.fft(y, axis=1))
    Nt = np.mean(y_a2[:NIS, ], axis=0)

    y_a2 = np.where(y_a2 >= a * Nt, y_a2 - a * Nt, b * Nt)

    X = y_a2 * np.cos(y_angle) + 1j * y_a2 * np.sin(y_angle)
    hatx = np.real(np.fft.ifft(X, axis=1))

    sig = np.zeros(int((fn - 1) * inc + wlen))

    for i in range(fn):
        start = i * inc
        sig[start:start + flen] += hatx[i, :]
    return sig


def SpectralSubIm(signal, wind, inc, NIS, Gamma, Beta):
    pass
