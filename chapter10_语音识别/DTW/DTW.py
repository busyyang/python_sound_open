import numpy as np
from chapter3_分析实验.timefeature import *
from chapter2_基础.soundBase import *
from chapter3_分析实验.mel import *


def my_vad(x):
    """
    端点检测
    :param X:输入为录入语音
    :return:输出为有用信号
    """
    Ini = 0.1  # 初始静默时间
    Ts = 0.01  # 窗的时长
    Tsh = 0.005  # 帧移时长
    Fs = 16000  # 采样频率
    counter1 = 0  # 以下四个参数用来寻找起始点和结束点
    counter2 = 0
    counter3 = 0
    counter4 = 0
    ZCRCountf = 0  # 用于存储过零率检测结果
    ZCRCountb = 0
    ZTh = 40  # 过零阈值
    w_sam = int(Ts * Fs)  # 窗口长度
    o_sam = int(Tsh * Fs)  # 帧移长度
    lengthX = len(x)
    segs = int((lengthX - w_sam) / o_sam) + 1
    sil = int((Ini - Ts) / Tsh) + 1
    win = np.hamming(w_sam)
    Limit = o_sam * (segs - 1) + 1
    FrmtIndex = [i for i in range(0, Limit, o_sam)]  # 每一帧的起始位置
    # 短时过零率
    ZCR_Vector = STZcr(x, win, o_sam)
    # 能量
    Erg_Vector = STMn(x, win, o_sam)
    IMN = np.mean(Erg_Vector[:sil])
    IMX = np.max(Erg_Vector)
    l1 = 0.03 * (IMX - IMN) + IMN
    l2 = 4 * IMN
    ITL = 100 * np.min((l1, l2))
    ITU = 10 * ITL
    IZC = np.mean(ZCR_Vector[:sil])
    stddev = np.std(ZCR_Vector[:sil])
    IZCT = np.min((ZTh, IZC + 2 * stddev))
    indexi = np.zeros(lengthX)
    indexj, indexk, indexl = indexi, indexi, indexi
    # 搜寻超过能量阈值上限的部分
    for i in range(len(Erg_Vector)):
        if Erg_Vector[i] > ITU:
            indexi[counter1] = i
            counter1 += 1
    ITUs = int(indexi[0])
    # 搜寻能量超过能量下限的部分
    for j in range(ITUs - 1, -1, -1):
        if Erg_Vector[j] < ITL:
            indexj[counter2] = j
            counter2 += 1
    start = int(indexj[0]) + 1
    Erg_Vectorf = np.flip(Erg_Vector, axis=0)
    # 重复上面过程相当于找结束帧
    for k in range(len(Erg_Vectorf)):
        if Erg_Vectorf[k] > ITU:
            indexi[counter3] = k
            counter3 += 1
    ITUs = int(indexk[0])
    for l in range(ITUs - 1, -1, -1):
        if Erg_Vectorf[l] < ITL:
            indexl[counter4] = l
            counter4 += 1
    finish = len(Erg_Vector) - int(indexl[0])  # 第一级判决结束帧
    # 从第一级判决起始帧开始进行第二判决（过零率）端点检测
    BackSearch = np.min((start, 25))
    for m in range(start, start - BackSearch, -1):
        rate = ZCR_Vector[m]
        if rate > IZCT:
            ZCRCountb += 1
            realstart = m
    if ZCRCountb > 3:
        start = realstart

    FwdSearch = np.min((len(Erg_Vector) - finish, 25))
    for n in range(finish, finish + FwdSearch):
        rate = ZCR_Vector[n]
        if rate > IZCT:
            ZCRCountf += 1
            realfinish = n
    if ZCRCountf > 3:
        finish = realfinish
    x_start = FrmtIndex[start]
    x_finish = FrmtIndex[finish]
    trimmed_X = x[x_start:x_finish]
    return trimmed_X


def myDCW(F, R):
    """
    动态时间规划
    :param F:为模板MFCC参数矩阵
    :param R:为当前语音MFCC参数矩阵
    :return:cost为最佳匹配距离
    """
    r1, c1 = F.shape
    r2, c2 = R.shape
    distence = np.zeros((r1, r2))
    for n in range(r1):
        for m in range(r2):
            FR = np.power(F[n, :] - R[m, :], 2)
            distence[n, m] = np.sqrt(np.sum(FR)) / c1

    D = np.zeros((r1 + 1, r2 + 1))
    D[0, :] = np.inf
    D[:, 0] = np.inf
    D[0, 0] = 0
    D[1:, 1:] = distence

    # 寻找整个过程的最短匹配距离
    for i in range(r1):
        for j in range(r2):
            dmin = min(D[i, j], D[i, j + 1], D[i + 1, j])
            D[i + 1, j + 1] += dmin

    cost = D[r1, r2]
    return cost


def deltacoeff(x):
    """
    计算MFCC差分系数
    :param x:
    :return:
    """
    nr, nc = x.shape
    N = 2
    diff = np.zeros((nr, nc))
    for t in range(2, nr - 2):
        for n in range(N):
            diff[t, :] += n * (x[t + n, :] - x[t - n, :])
        diff[t, :] /= 10
    return diff


def mfccf(num, s, Fs):
    """
    计算并返回信号s的mfcc参数及其一阶和二阶差分参数
    :param num:
    :param s:
    :param Fs:
    :return:
    """
    N = 512  # FFT数
    Tf = 0.02  # 窗口的时长
    n = int(Fs * Tf)  # 每个窗口的长度
    M = 24  # M为滤波器组数
    l = len(s)
    Ts = 0.01  # 帧移时长
    FrameStep = int(Fs * Ts)  # 帧移
    lifter = np.array([i for i in range(num)])
    lifter = 1 + int(num / 2) * np.sin(lifter * np.pi / num)

    if np.mean(np.abs(s)) > 0.01:
        s = s / np.max(s)
    # 计算MFCC
    mfcc = Nmfcc(s, Fs, M, n, FrameStep)

    mfcc_l = np.multiply(mfcc, lifter)
    d1 = deltacoeff(mfcc_l)
    d2 = deltacoeff(d1)
    return np.hstack((mfcc_l, d1, d2))


def CMN(r):
    """
    归一化
    :param r:
    :return:
    """
    return r - np.mean(r, axis=1, keepdims=True)


def DTWScores(r, features, N):
    """
    DTW寻找最小失真值
    :param r:为当前读入语音的MFCC参数矩阵
    :param features:模型参数
    :param N:为每个模板数量词汇数
    :return:
    """
    # 初始化判别矩阵
    scores1 = np.zeros(N)
    scores2 = np.zeros(N)
    scores3 = np.zeros(N)

    for i in range(N):
        scores1[i] = myDCW(CMN(features['p1_{}'.format(i)]), r)
        scores2[i] = myDCW(CMN(features['p2_{}'.format(i)]), r)
        scores3[i] = myDCW(CMN(features['p2_{}'.format(i)]), r)
    return scores1, scores2, scores3


if __name__ == '__main__':
    # 制作模板集
    features = {}
    for i in range(10):
        data, fs = soundBase('p1/{}.wav'.format(i)).audioread()
        speechIn1 = my_vad(data)
        fm = mfccf(12, speechIn1, fs)
        features['p1_{}'.format(i)] = fm
    for i in range(10):
        data, fs = soundBase('p2/{}.wav'.format(i)).audioread()
        speechIn1 = my_vad(data)
        fm = mfccf(12, speechIn1, fs)
        features['p2_{}'.format(i)] = fm
    for i in range(10):
        data, fs = soundBase('p3/{}.wav'.format(i)).audioread()
        speechIn1 = my_vad(data)
        fm = mfccf(12, speechIn1, fs)
        features['p3_{}'.format(i)] = fm

    soundBase('mysound.wav').audiorecorder(rate=16000, channels=1)
    data, fs = soundBase('mysound.wav').audioread()
    mysound = my_vad(data)
    fm_my = mfccf(12, mysound, fs)
    fm_my = CMN(fm_my)

    scores = DTWScores(fm_my, features, 10)
    s = {}
    for i in range(len(scores)):
        tmp = np.argsort(scores[i])[:2]
        for j in tmp:
            if j in s.keys():
                s[j] += 1
            else:
                s[j] = 1
    print(s)
    s = sorted(s.items(), key=lambda x: x[1], reverse=True)[0]

    if np.max(data) < 0.01:
        print('Microphone get no signal......')
    elif s[1] < 2:
        print('The word you have said could not be properly recognised.')
    else:
        print('You just saied: {}'.format(s[0]))
