# 共振峰估计函数
import numpy as np
from chapter3_分析实验.timefeature import *
from chapter3_分析实验.lpc import lpc_coeff


def local_maxium(x):
    """
    求序列的极大值
    :param x:
    :return:
    """
    d = np.diff(x)
    l_d = len(d)
    maxium = []
    loc = []
    for i in range(l_d - 1):
        if d[i] > 0 and d[i + 1] <= 0:
            maxium.append(x[i + 1])
            loc.append(i + 1)
    return maxium, loc


def Formant_Cepst(u, cepstL):
    """
    倒谱法共振峰估计函数
    :param u:
    :param cepstL:
    :return:
    """
    wlen2 = len(u) // 2
    U = np.log(np.abs(np.fft.fft(u)[:wlen2]))
    Cepst = np.fft.ifft(U)
    cepst = np.zeros(wlen2, dtype=np.complex)
    cepst[:cepstL] = Cepst[:cepstL]
    cepst[-cepstL + 1:] = Cepst[-cepstL + 1:]
    spec = np.real(np.fft.fft(cepst))
    val, loc = local_maxium(spec)
    return val, loc, spec


def Formant_Interpolation(u, p, fs):
    """
    插值法估计共振峰函数
    :param u:
    :param p:
    :param fs:
    :return:
    """
    ar, _ = lpc_coeff(u, p)
    U = np.power(np.abs(np.fft.rfft(ar, 2 * 255)), -2)
    df = fs / 512
    val, loc = local_maxium(U)
    ll = len(loc)
    pp = np.zeros(ll)
    F = np.zeros(ll)
    Bw = np.zeros(ll)
    for k in range(ll):
        m = loc[k]
        m1, m2 = m - 1, m + 1
        p = val[k]
        p1, p2 = U[m1], U[m2]
        aa = (p1 + p2) / 2 - p
        bb = (p2 - p1) / 2
        cc = p
        dm = -bb / 2 / aa
        pp[k] = -bb * bb / 4 / aa + cc
        m_new = m + dm
        bf = -np.sqrt(bb * bb - 4 * aa * (cc - pp[k] / 2)) / aa
        F[k] = (m_new - 1) * df
        Bw[k] = bf * df
    return F, Bw, pp, U, loc


def Formant_Root(u, p, fs, n_frmnt):
    """
    LPC求根法的共振峰估计函数
    :param u:
    :param p:
    :param fs:
    :param n_frmnt:
    :return:
    """
    ar, _ = lpc_coeff(u, p)
    U = np.power(np.abs(np.fft.rfft(ar, 2 * 255)), -2)
    const = fs / (2 * np.pi)
    rts = np.roots(ar)
    yf = []
    Bw = []
    for i in range(len(ar) - 1):
        re = np.real(rts[i])
        im = np.imag(rts[i])
        fromn = const * np.arctan2(im, re)
        bw = -2 * const * np.log(np.abs(rts[i]))
        if fromn > 150 and bw < 700 and fromn < fs / 2:
            yf.append(fromn)
            Bw.append(bw)
    return yf[:min(len(yf), n_frmnt)], Bw[:min(len(Bw), n_frmnt)], U
