import numpy as np


def Filpframe_OverlapA(x, win, inc):
    """
    基于重叠相加法的信号还原函数
    :param x: 分帧数据
    :param win: 窗
    :param inc: 帧移
    :return:
    """
    nf, slen = x.shape
    nx = (nf - 1) * inc + slen
    frameout = np.zeros(nx)
    x = x / win
    for i in range(nf):
        start = i * inc
        frameout[start:start + slen] += x[i, :]
    return frameout


def Filpframe_OverlapS(x, win, inc):
    """
    基于重叠存储法的信号还原函数
    :param x: 分帧数据
    :param win: 窗
    :param inc: 帧移
    :return:
    """
    nf, slen = x.shape
    nx = (nf - 1) * inc + slen
    frameout = np.zeros(nx)
    x = x / win
    for i in range(nf):
        frameout[slen + (i - 1) * inc:slen + i * inc] += x[i, slen - inc:]
    return frameout


def Filpframe_LinearA(x, win, inc):
    """
    基于比例重叠相加法的信号还原函数
    :param x: 分帧数据
    :param win: 窗
    :param inc: 帧移
    :return:
    """
    nf, slen = x.shape
    nx = (nf - 1) * inc + slen
    frameout = np.zeros(nx)
    overlap = len(win) - inc
    x = x / win
    w1 = [i / overlap for i in range(overlap)]
    w2 = [i / overlap for i in range(overlap - 1, -1, -1)]
    for i in range(nf):
        if i == 0:
            frameout[:slen] = x[i, :]
        else:
            M = slen + (i - 1) * inc
            y = frameout[M - overlap:M] * w2 + x[i, :overlap] * w1
            xn = x[i, overlap:]
            yy = np.hstack((y, xn))
            frameout[M - overlap:M - overlap + slen] += yy
    return frameout
