from chapter3_分析实验.C3_1_y_1 import enframe
from chapter3_分析实验.timefeature import *


def vad_revr(dst1, T1, T2):
    """
    端点检测反向比较函数
    :param dst1:
    :param T1:
    :param T2:
    :return:
    """
    fn = len(dst1)
    maxsilence = 8
    minlen = 5
    status = 0
    count = np.zeros(fn)
    silence = np.zeros(fn)
    xn = 0
    x1 = np.zeros(fn)
    x2 = np.zeros(fn)
    for n in range(1, fn):
        if status == 0 or status == 1:
            if dst1[n] < T2:
                x1[xn] = max(1, n - count[xn] - 1)
                status = 2
                silence[xn] = 0
                count[xn] += 1
            elif dst1[n] < T1:
                status = 1
                count[xn] += 1
            else:
                status = 0
                count[xn] = 0
                x1[xn] = 0
                x2[xn] = 0
        if status == 2:
            if dst1[n] < T1:
                count[xn] += 1
            else:
                silence[xn] += 1
                if silence[xn] < maxsilence:
                    count[xn] += 1
                elif count[xn] < minlen:
                    status = 0
                    silence[xn] = 0
                    count[xn] = 0
                else:
                    status = 3
                    x2[xn] = x1[xn] + count[xn]
        if status == 3:
            status = 0
            xn += 1
            count[xn] = 0
            silence[xn] = 0
            x1[xn] = 0
            x2[xn] = 0
    el = len(x1[:xn])
    if x1[el - 1] == 0:
        el -= 1
    if x2[el - 1] == 0:
        print('Error: Not find endding point!\n')
        x2[el] = fn
    SF = np.zeros(fn)
    NF = np.ones(fn)
    for i in range(el):
        SF[int(x1[i]):int(x2[i])] = 1
        NF[int(x1[i]):int(x2[i])] = 0
    voiceseg = findSegment(np.where(SF == 1)[0])
    vsl = len(voiceseg.keys())
    return voiceseg, vsl, SF, NF


def vad_forw(dst1, T1, T2):
    """
    端点检测正向比较函数
    :param dst1:
    :param T1:
    :param T2:
    :return:
    """
    fn = len(dst1)
    maxsilence = 8
    minlen = 5
    status = 0
    count = np.zeros(fn)
    silence = np.zeros(fn)
    xn = 0
    x1 = np.zeros(fn)
    x2 = np.zeros(fn)
    for n in range(1, fn):
        if status == 0 or status == 1:
            if dst1[n] > T2:
                x1[xn] = max(1, n - count[xn] - 1)
                status = 2
                silence[xn] = 0
                count[xn] += 1
            elif dst1[n] > T1:
                status = 1
                count[xn] += 1
            else:
                status = 0
                count[xn] = 0
                x1[xn] = 0
                x2[xn] = 0
        if status == 2:
            if dst1[n] > T1:
                count[xn] += 1
            else:
                silence[xn] += 1
                if silence[xn] < maxsilence:
                    count[xn] += 1
                elif count[xn] < minlen:
                    status = 0
                    silence[xn] = 0
                    count[xn] = 0
                else:
                    status = 3
                    x2[xn] = x1[xn] + count[xn]
        if status == 3:
            status = 0
            xn += 1
            count[xn] = 0
            silence[xn] = 0
            x1[xn] = 0
            x2[xn] = 0
    el = len(x1[:xn])
    if x1[el - 1] == 0:
        el -= 1
    if x2[el - 1] == 0:
        print('Error: Not find endding point!\n')
        x2[el] = fn
    SF = np.zeros(fn)
    NF = np.ones(fn)
    for i in range(el):
        SF[int(x1[i]):int(x2[i])] = 1
        NF[int(x1[i]):int(x2[i])] = 0
    voiceseg = findSegment(np.where(SF == 1)[0])
    vsl = len(voiceseg.keys())
    return voiceseg, vsl, SF, NF


def findSegment(express):
    """
    分割成語音段
    :param express:
    :return:
    """
    if express[0] == 0:
        voiceIndex = np.where(express)
    else:
        voiceIndex = express
    d_voice = np.where(np.diff(voiceIndex) > 1)[0]
    voiceseg = {}
    if len(d_voice) > 0:
        for i in range(len(d_voice) + 1):
            seg = {}
            if i == 0:
                st = voiceIndex[0]
                en = voiceIndex[d_voice[i]]
            elif i == len(d_voice):
                st = voiceIndex[d_voice[i - 1] + 1]
                en = voiceIndex[-1]
            else:
                st = voiceIndex[d_voice[i - 1] + 1]
                en = voiceIndex[d_voice[i]]
            seg['start'] = st
            seg['end'] = en
            seg['duration'] = en - st + 1
            voiceseg[i] = seg
    return voiceseg


def vad_TwoThr(x, wlen, inc, NIS):
    """
    使用门限法检测语音段
    :param x: 语音信号
    :param wlen: 分帧长度
    :param inc: 帧移
    :param NIS:
    :return:
    """
    maxsilence = 15
    minlen = 5
    status = 0
    y = enframe(x, wlen, inc)
    fn = y.shape[0]
    amp = STEn(x, wlen, inc)
    zcr = STZcr(x, wlen, inc, delta=0.01)
    ampth = np.mean(amp[:NIS])
    zcrth = np.mean(zcr[:NIS])
    amp2 = 2 * ampth
    amp1 = 4 * ampth
    zcr2 = 2 * zcrth
    xn = 0
    count = np.zeros(fn)
    silence = np.zeros(fn)
    x1 = np.zeros(fn)
    x2 = np.zeros(fn)
    for n in range(fn):
        if status == 0 or status == 1:
            if amp[n] > amp1:
                x1[xn] = max(1, n - count[xn] - 1)
                status = 2
                silence[xn] = 0
                count[xn] += 1
            elif amp[n] > amp2 or zcr[n] > zcr2:
                status = 1
                count[xn] += 1
            else:
                status = 0
                count[xn] = 0
                x1[xn] = 0
                x2[xn] = 0

        elif status == 2:
            if amp[n] > amp2 and zcr[n] > zcr2:
                count[xn] += 1
            else:
                silence[xn] += 1
                if silence[xn] < maxsilence:
                    count[xn] += 1
                elif count[xn] < minlen:
                    status = 0
                    silence[xn] = 0
                    count[xn] = 0
                else:
                    status = 3
                    x2[xn] = x1[xn] + count[xn]
        elif status == 3:
            status = 0
            xn += 1
            count[xn] = 0
            silence[xn] = 0
            x1[xn] = 0
            x2[xn] = 0
    el = len(x1[:xn])
    if x1[el - 1] == 0:
        el -= 1
    if x2[el - 1] == 0:
        print('Error: Not find endding point!\n')
        x2[el] = fn
    SF = np.zeros(fn)
    NF = np.ones(fn)
    for i in range(el):
        SF[int(x1[i]):int(x2[i])] = 1
        NF[int(x1[i]):int(x2[i])] = 0
    voiceseg = findSegment(np.where(SF == 1)[0])
    vsl = len(voiceseg.keys())
    return voiceseg, vsl, SF, NF, amp, zcr


def vad_corr(y, wnd, inc, NIS, th1, th2):
    x = enframe(y, wnd, inc)
    Ru = STAc(x.T)[0]
    Rum = Ru / np.max(Ru)
    thredth = np.max(Rum[:NIS])
    T1 = th1 * thredth
    T2 = th2 * thredth
    voiceseg, vsl, SF, NF = vad_forw(Rum, T1, T2)
    return voiceseg, vsl, SF, NF, Rum


def vad_specEN(data, wnd, inc, NIS, thr1, thr2, fs):
    from scipy.signal import medfilt
    x = enframe(data, wnd, inc)
    X = np.abs(np.fft.fft(x, axis=1))
    if len(wnd) == 1:
        wlen = wnd
    else:
        wlen = len(wnd)
    df = fs / wlen
    fx1 = int(250 // df + 1)  # 250Hz位置
    fx2 = int(3500 // df + 1)  # 500Hz位置
    km = wlen // 8
    K = 0.5
    E = np.zeros((X.shape[0], wlen // 2))
    E[:, fx1 + 1:fx2 - 1] = X[:, fx1 + 1:fx2 - 1]
    E = np.multiply(E, E)
    Esum = np.sum(E, axis=1, keepdims=True)
    P1 = np.divide(E, Esum)
    E = np.where(P1 >= 0.9, 0, E)
    Eb0 = E[:, 0::4]
    Eb1 = E[:, 1::4]
    Eb2 = E[:, 2::4]
    Eb3 = E[:, 3::4]
    Eb = Eb0 + Eb1 + Eb2 + Eb3
    prob = np.divide(Eb + K, np.sum(Eb + K, axis=1, keepdims=True))
    Hb = -np.sum(np.multiply(prob, np.log10(prob + 1e-10)), axis=1)
    for i in range(10):
        Hb = medfilt(Hb, 5)
    Me = np.mean(Hb)
    eth = np.mean(Hb[:NIS])
    Det = eth - Me
    T1 = thr1 * Det + Me
    T2 = thr2 * Det + Me
    voiceseg, vsl, SF, NF = vad_revr(Hb, T1, T2)
    return voiceseg, vsl, SF, NF, Hb


def vad_pro(data, wnd, inc, NIS, thr1, thr2, mode):
    """
    使用比例法检测端点
    :param data:
    :param wnd:
    :param inc:
    :param NIS:
    :param thr1:
    :param thr2:
    :param mode:
    :return:
    """
    from scipy.signal import medfilt
    x = enframe(data, wnd, inc)
    if len(wnd) == 1:
        wlen = wnd
    else:
        wlen = len(wnd)
    if mode == 1:  # 能零比
        a = 2
        b = 1
        LEn = np.log10(1 + np.sum(np.multiply(x, x) / a, axis=1))
        EZRn = LEn / (STZcr(data, wlen, inc) + b)
        for i in range(10):
            EZRn = medfilt(EZRn, 5)
        dth = np.mean(EZRn[:NIS])
        T1 = thr1 * dth
        T2 = thr2 * dth
        Epara = EZRn
    elif mode == 2:  # 能熵比
        a = 2
        X = np.abs(np.fft.fft(x, axis=1))
        X = X[:, :wlen // 2]
        Esum = np.log10(1 + np.sum(np.multiply(X, X) / a, axis=1))
        prob = X / np.sum(X, axis=1, keepdims=True)
        Hn = -np.sum(np.multiply(prob, np.log10(prob + 1e-10)), axis=1)
        Ef = np.sqrt(1 + np.abs(Esum / Hn))
        for i in range(10):
            Ef = medfilt(Ef, 5)
        Me = np.max(Ef)
        eth = np.mean(Ef[NIS])
        Det = Me - eth
        T1 = thr1 * Det + eth
        T2 = thr2 * Det + eth
        Epara = Ef
    voiceseg, vsl, SF, NF = vad_forw(Epara, T1, T2)
    return voiceseg, vsl, SF, NF, Epara


def vad_LogSpec(signal, noise, NoiseCounter=0, NoiseMargin=3, Hangover=8):
    """
    对数频率距离检测语音端点
    :param signal:
    :param noise:
    :param NoiseCounter:
    :param NoiseMargin:
    :param Hangover:
    :return:
    """
    SpectralDist = 20 * (np.log10(signal) - np.log10(noise))
    SpectralDist = np.where(SpectralDist < 0, 0, SpectralDist)
    Dist = np.mean(SpectralDist)
    if Dist < NoiseMargin:
        NoiseFlag = 1
        NoiseCounter += 1
    else:
        NoiseFlag = 0
        NoiseCounter = 0
    if NoiseCounter > Hangover:
        SpeechFlag = 0
    else:
        SpeechFlag = 1
    return NoiseFlag, SpeechFlag, NoiseCounter, Dist
