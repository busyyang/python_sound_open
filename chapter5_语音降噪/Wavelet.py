import pywt
import numpy as np

from chapter3_分析实验.mel import *


def greyList(n):
    """
    生成格雷编码序列
    参考：https://www.jb51.net/article/133575.htm
    :param n: 长度
    :return: 范围 2 ** n的格雷序列
    """

    def get_grace(list_grace, n):
        if n == 1:
            return list_grace
        list_before, list_after = [], []
        for i in range(len(list_grace)):
            list_before.append('0' + list_grace[i])
            list_after.append('1' + list_grace[-(i + 1)])
        return get_grace(list_before + list_after, n - 1)

    # get_grace生成的序列是二进制字符串，转化为10进制数
    return [int(i, 2) for i in get_grace(['0', '1'], n)]


def wavePacketRec(s, wname):
    """
    小波包重构
    :param s: 小波包分解系数，(a,d)间隔
    :param wname: 小波名
    :return: 各个分量的重构,注意这里重构的长度可能稍微大于原来的长度，如果要保证一样长，去除尾段的即可
    """
    out = []
    for i, l in enumerate(s):
        # 利用该数字去判断是a,d分量类型
        magic = i
        for j in range(int(np.log2(len(s)))):
            if magic % 2 == 0:
                # 为a分量时
                l = pywt.waverec([l, None], wname)
            else:
                # 为d分量时
                l = pywt.waverec([None, l], wname)
            magic = magic // 2
        out.append(l)

    grey_order = greyList(int(np.log2(len(s))))
    return np.array(out)[grey_order]


def wavePacketDec(s, jN, wname):
    """
    小波包分解，分解得到的是最后一层的结果，并非小波系数
    :param s: 源信号
    :param jN: 分解层数
    :param wname: 小波名
    :return:
        out: 小波包系数,是len为2**jN的list,以a,d相互间隔的输出
    """
    assert jN > 0 and isinstance(jN, int), 'Please take a positive integer as jN'
    out = []
    # 第一层分解
    a, d = pywt.dwt(s, wname)
    out.append(a)
    out.append(d)

    # 执行第2-jN次分解
    for level in range(1, jN):
        tmp = []
        for i in range(2 ** level):
            a, d = pywt.dwt(out[i], wname)
            tmp.append(a)
            tmp.append(d)
        out = tmp
    return out


def WPMFCC(x, fs, p, frameSize, inc, jN, wname, nFFT=512, n_dct=12):
    """
    利用小波包-MFCC的方式提取特征，处理流程参考：
    陈静，基于小波包变换和MFCC的说话人识别特征参数，语音技术，2009，DOI:10.16311/j.audioe.2009.02.017
    :param x: 输入信号
    :param fs: 采样率
    :param p: Mel滤波器组的个数
    :param frameSize: 分帧的每帧长度
    :param inc: 帧移
    :param jN: 小波包分解尺度
    :param wname: 小波包名
    :param nFFT: FFT点数
    :param n_dct: DCT阶数
    :return:
    """
    # 预处理-预加重
    xx = lfilter([1, -0.9375], [1], x)
    # 预处理-分幀
    xx = enframe(xx, frameSize, inc)
    # 预处理-加窗
    xx = np.multiply(xx, np.hanning(frameSize))
    # 分层计算小波包能量谱线并合并
    enp = np.zeros((xx.shape[0], nFFT // 2 + 1))
    for i, l in enumerate(xx):
        # 进行小波包分解
        wpcoeff = wavePacketDec(l, jN, wname)
        wp_l = wavePacketRec(wpcoeff, wname)[:, :frameSize]
        # 计算谱线能量
        en = np.abs(np.fft.rfft(wp_l, nFFT)) ** 2 / nFFT
        # 谱线合并
        enp[i] = np.sum(en, axis=0)
    # 计算通过Mel滤波器的能量
    bank = melbankm(p, nFFT, fs, 0, 0.5 * fs, 0)
    ss = np.matmul(enp, bank.T)
    # 计算DCT倒谱
    M = bank.shape[0]
    m = np.array([i for i in range(M)])
    mfcc = np.zeros((ss.shape[0], n_dct))
    for n in range(n_dct):
        mfcc[:, n] = np.sqrt(2 / M) * np.sum(np.multiply(np.log(ss), np.cos((2 * m - 1) * n * np.pi / 2 / M)), axis=1)
    return mfcc


def wavedec(s, jN, wname):
    ca, cd = [], []
    a = s
    for i in range(jN):
        a, d = pywt.dwt(a, wname)
        ca.append(a)
        cd.append(d)
    return ca, cd


def Wavelet_Hard(s, jN, wname):
    """
    小波硬阈值滤波
    :param s:
    :param jN:
    :param wname:
    :return:
    """
    ca, cd = wavedec(s, jN, wname)
    for i in range(len(ca)):
        thr = np.median(cd[i] * np.sqrt(2 * np.log((i + 2) / (i + 1)))) / 0.6745
        di = np.array(cd[i])
        cd[i] = np.where(np.abs(di) > thr, di, 0)
    calast = np.array(ca[-1])
    thr = np.median(calast * np.sqrt(2 * np.log((jN + 1) / jN))) / 0.6745
    calast = np.where(np.abs(calast) > thr, di, 0)
    cd.append(calast)
    coef = cd[::-1]
    res = pywt.waverec(coef, wname)
    return res


def Wavelet_Soft(s, jN, wname):
    """
    小波软阈值滤波
    :param s:
    :param jN:
    :param wname:
    :return:
    """
    ca, cd = wavedec(s, jN, wname)
    for i in range(len(ca)):
        thr = np.median(cd[i] * np.sqrt(2 * np.log((i + 2) / (i + 1)))) / 0.6745
        di = np.array(cd[i])
        cd[i] = np.where(np.abs(di) > thr, np.sign(di) * (np.abs(di) - thr), 0)
    calast = np.array(ca[-1])
    thr = np.median(calast * np.sqrt(2 * np.log((jN + 1) / jN))) / 0.6745
    calast = np.where(np.abs(calast) > thr, np.sign(calast) * (np.abs(calast) - thr), 0)
    cd.append(calast)
    coef = cd[::-1]
    res = pywt.waverec(coef, wname)
    return res


def Wavelet_hardSoft(s, jN, wname, alpha=0.5):
    """
    小波折中阈值滤波
    :param s:
    :param jN:
    :param wname:
    :param alpha:
    :return:
    """
    ca, cd = wavedec(s, jN, wname)
    for i in range(len(ca)):
        thr = np.median(cd[i] * np.sqrt(2 * np.log((i + 2) / (i + 1)))) / 0.6745
        di = np.array(cd[i])
        cd[i] = np.where(np.abs(di) > thr, np.sign(di) * (np.abs(di) - alpha * thr), 0)
    calast = np.array(ca[-1])
    thr = np.median(calast * np.sqrt(2 * np.log((jN + 1) / jN))) / 0.6745
    calast = np.where(np.abs(calast) > thr, np.sign(calast) * (np.abs(calast) - alpha * thr), 0)
    cd.append(calast)
    coef = cd[::-1]
    res = pywt.waverec(coef, wname)
    return res


def Wavelet_average(s, jN, wname, mu=0.1):
    """
    小波加权平均滤波
    :param s:
    :param jN:
    :param wname:
    :param alpha:
    :return:
    """
    ca, cd = wavedec(s, jN, wname)
    for i in range(len(ca)):
        thr = np.median(cd[i] * np.sqrt(2 * np.log((i + 2) / (i + 1)))) / 0.6745
        di = np.array(cd[i])
        cd[i] = np.where(np.abs(di) > thr, (1 - mu) * di + np.sign(di) * mu * (np.abs(di) - thr), 0)
    calast = np.array(ca[-1])
    thr = np.median(calast * np.sqrt(2 * np.log((jN + 1) / jN))) / 0.6745
    calast = np.where(np.abs(calast) > thr, (1 - mu) * calast + np.sign(calast) * mu * (np.abs(calast) - thr), 0)
    cd.append(calast)
    coef = cd[::-1]
    res = pywt.waverec(coef, wname)
    return res
