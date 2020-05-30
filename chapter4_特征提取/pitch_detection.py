from chapter3_分析实验.C3_1_y_1 import enframe
from chapter3_分析实验.timefeature import *
from chapter4_特征提取.end_detection import findSegment


def pitch_vad(x, wnd, inc, T1, miniL=10):
    """
    使用能熵比检测基音，实际上就是语音分段
    :param x:
    :param wnd:
    :param inc:
    :param T1:
    :param miniL:
    :return:
    """
    y = enframe(x, wnd, inc)
    fn = y.shape[0]
    if isinstance(wnd, int):
        wlen = wnd
    else:
        wlen = len(wnd)

    Sp = np.abs(np.fft.fft(y, axis=1))
    Sp = Sp[:, :wlen // 2 + 1]
    Esum = np.sum(np.multiply(Sp, Sp), axis=1)
    prob = Sp / np.sum(Sp, axis=1, keepdims=True)
    H = -np.sum(np.multiply(prob, np.log10(prob + 1e-16)), axis=1)
    H = np.where(H < 0.1, np.max(H), H)
    Ef = np.sqrt(1 + np.abs(Esum / H))
    Ef = Ef / np.max(Ef)

    zseg = findSegment(np.where(Ef > T1)[0])
    zsl = len(zseg.keys())
    SF = np.zeros(fn)
    for k in range(zsl):
        if zseg[k]['duration'] < miniL:
            zseg.pop(k)
        else:
            SF[zseg[k]['start']:zseg[k]['end']] = 1
    return zseg, len(zseg.keys()), SF, Ef


def pitch_Ceps(x, wnd, inc, T1, fs, miniL=10):
    """
    倒谱法基音周期检测函数
    :param x:
    :param wnd:
    :param inc:
    :param T1:
    :param fs:
    :param miniL:
    :return:
    """
    y = enframe(x, wnd, inc)
    fn = y.shape[0]
    if isinstance(wnd, int):
        wlen = wnd
    else:
        wlen = len(wnd)
    voiceseg, vsl, SF, Ef = pitch_vad(x, wnd, inc, T1, miniL)
    lmin = fs // 500  # 基音周期的最小值
    lmax = fs // 60  # 基音周期的最大值
    period = np.zeros(fn)
    y1 = y[np.where(SF == 1)[0], :]
    y1 = np.multiply(y1, np.hamming(wlen))
    xx = np.fft.fft(y1, axis=1)
    b = np.fft.ifft(2 * np.log(np.abs(xx) + 1e-10))
    Lc = np.argmax(b[:, lmin:lmax], axis=1) + lmin - 1
    period[np.where(SF == 1)[0]] = Lc
    return voiceseg, vsl, SF, Ef, period


def pitch_Corr(x, wnd, inc, T1, fs, miniL=10):
    """
    自相关法基音周期检测函数
    :param x: 
    :param wnd: 
    :param inc: 
    :param T1: 
    :param fs: 
    :param miniL: 
    :return: 
    """
    y = enframe(x, wnd, inc)
    fn = y.shape[0]
    if isinstance(wnd, int):
        wlen = wnd
    else:
        wlen = len(wnd)
    voiceseg, vsl, SF, Ef = pitch_vad(x, wnd, inc, T1, miniL)
    lmin = fs // 500  # 基音周期的最小值
    lmax = fs // 60  # 基音周期的最大值
    period = np.zeros(fn)
    for i in range(vsl):
        ixb = voiceseg[i]['start']
        ixd = voiceseg[i]['duration']
        for k in range(ixd):
            ru = np.correlate(y[k + ixb, :], y[k + ixb, :], 'full')
            ru = ru[wlen:]
            tloc = np.argmax(ru[lmin:lmax])
            period[k + ixb] = lmin + tloc

    return voiceseg, vsl, SF, Ef, period


def pitch_Lpc(x, wnd, inc, T1, fs, p, miniL=10):
    """
    线性预测法基音周期检测函数
    :param x:
    :param wnd:
    :param inc:
    :param T1:
    :param fs:
    :param p:
    :param miniL:
    :return:
    """
    from scipy.signal import lfilter
    from chapter3_分析实验.lpc import lpc_coeff
    y = enframe(x, wnd, inc)
    fn = y.shape[0]
    if isinstance(wnd, int):
        wlen = wnd
    else:
        wlen = len(wnd)
    voiceseg, vsl, SF, Ef = pitch_vad(x, wnd, inc, T1, miniL)
    lmin = fs // 500  # 基音周期的最小值
    lmax = fs // 60  # 基音周期的最大值
    period = np.zeros(fn)
    for k in range(y.shape[0]):
        if SF[k] == 1:
            u = np.multiply(y[k, :], np.hamming(wlen))
            ar, _ = lpc_coeff(u, p)
            ar[0] = 0
            z = lfilter(-ar, [1], u)
            E = u - z
            xx = np.fft.fft(E)
            b = np.fft.ifft(2 * np.log(np.abs(xx) + 1e-20))
            lc = np.argmax(b[lmin:lmax])
            period[k] = lc + lmin
    return voiceseg, vsl, SF, Ef, period
