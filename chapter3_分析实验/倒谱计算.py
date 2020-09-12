import numpy as np


def cceps(x):
    """
    计算复倒谱
    """
    y = np.fft.fft(x)
    return np.fft.ifft(np.log(y))


def icceps(y):
    """
    计算复倒谱的逆变换
    """
    x = np.fft.fft(y)
    return np.fft.ifft(np.exp(x))


def rcceps(x):
    """
    计算实倒谱
    """
    y = np.fft.fft(x)
    return np.fft.ifft(np.log(np.abs(y)))
