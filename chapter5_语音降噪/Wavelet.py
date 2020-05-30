import pywt
import numpy as np


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
