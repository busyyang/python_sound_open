import numpy as np
import pylab as plt
from scipy.signal import lfilter
from .C3_1_y_1 import enframe


def melbankm(p, NFFT, fs, fl, fh, w=None):
    """
    计算Mel滤波器组
    :param p: 滤波器个数
    :param n: 一帧FFT后的数据长度
    :param fs: 采样率
    :param fl: 最低频率
    :param fh: 最高频率
    :param w: 窗(没有加窗，无效参数)
    :return:
    """
    bl = 1125 * np.log(1 + fl / 700)  # 把 Hz 变成 Mel
    bh = 1125 * np.log(1 + fh / 700)
    B = bh - bl  # Mel带宽
    y = np.linspace(0, B, p + 2)  # 将梅尔刻度等间隔
    Fb = 700 * (np.exp(y / 1125) - 1)  # 把 Mel 变成Hz
    W2 = int(NFFT / 2 + 1)
    df = fs / NFFT
    freq = [int(i * df) for i in range(W2)]  # 采样频率值
    bank = np.zeros((p, W2))
    for k in range(1, p + 1):
        f0, f1, f2 = Fb[k], Fb[k - 1], Fb[k + 1]
        n1 = np.floor(f1 / df)
        n2 = np.floor(f2 / df)
        n0 = np.floor(f0 / df)
        for i in range(1, W2):
            if n1 <= i <= n0:
                bank[k - 1, i] = (i - n1) / (n0 - n1)
            elif n0 < i <= n2:
                bank[k - 1, i] = (n2 - i) / (n2 - n0)
            elif i > n2:
                break
        # plt.plot(freq, bank[k - 1, :], 'r')
    # plt.savefig('images/mel.png')
    return bank


def Nmfcc(x, fs, p, frameSize, inc, nfft=512, n_dct=12):
    """
    计算mfcc系数
    :param x: 输入信号
    :param fs: 采样率
    :param p: Mel滤波器组的个数
    :param frameSize: 分帧的每帧长度
    :param inc: 帧移
    :return:
    """
    # 预处理-预加重
    xx = lfilter([1, -0.9375], [1], x)
    # 预处理-分幀
    xx = enframe(xx, frameSize, inc)
    # 预处理-加窗
    xx = np.multiply(xx, np.hanning(frameSize))
    # 计算FFT
    xx = np.fft.rfft(xx, nfft)
    # 计算能量谱
    xx = (np.abs(xx) ** 2) / nfft
    # 计算通过Mel滤波器的能量
    bank = melbankm(p, nfft, fs, 0, 0.5 * fs, 0)
    ss = np.matmul(xx, bank.T)
    # 计算DCT倒谱
    M = bank.shape[0]
    m = np.array([i for i in range(M)])
    mfcc = np.zeros((ss.shape[0], n_dct))
    for n in range(n_dct):
        mfcc[:, n] = np.sqrt(2 / M) * np.sum(np.multiply(np.log(ss), np.cos((2 * m - 1) * n * np.pi / 2 / M)), axis=1)
    return mfcc


if __name__ == '__main__':
    melbankm(24, 256, 8000, 0, 4000)
