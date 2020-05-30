import numpy as np


def adpcm_decoder(code, sign_bit):
    """
    APDCM解码函数
    :param code:
    :param sign_bit:
    :return:
    """
    l = len(code)
    y = np.zeros(l)
    ss2 = np.zeros(l)
    ss2[0] = 1
    # 生成步长查找表
    index = [-1, 4]
    currentIndex = 0
    startval = 1
    endval = 127
    base = np.exp(np.log(2) / 8)
    # 近似步长
    const = startval / base
    numSteps = int(round(np.log(endval / const) / np.log(base)))
    n = [i + 1 for i in range(numSteps)]
    base = np.exp(np.log(endval / startval) / (numSteps - 1))
    const = startval / base
    tabel2 = np.round(const * np.power(base, n))

    for n in range(1, l):
        neg = code[n] >= sign_bit
        if neg:
            temp = code[n] - sign_bit
        else:
            temp = code[n]
        temp2 = (temp + 0.5) * ss2[n - 1]
        if neg:
            temp2 = -temp2
        y[n] = y[n - 1] + temp2
        if y[n] > 127:
            y[n] = 127
        elif y[n] < -127:
            y[n] = -127
        # 计算新的步长
        currentIndex += index[int(temp)]
        if currentIndex < 0:
            currentIndex = 0
        elif currentIndex > numSteps:
            currentIndex = numSteps
        ss2[n] = tabel2[currentIndex]
    return y / 128


def adpcm_encoder(x, sign_bit):
    """
    APDCM编码函数
    :param x:
    :param sign_bit:
    :return:
    """
    x *= 128
    l = len(x)
    # 生成步长查找表
    index = [-1, 4]
    currentIndex = 1
    startval = 1
    endval = 127
    base = np.exp(np.log(2) / 8)
    # 近似步长
    const = startval / base
    numSteps = int(round(np.log(endval / const) / np.log(base)))
    n = [i + 1 for i in range(numSteps)]
    base = np.exp(np.log(endval / startval) / (numSteps - 1))
    const = startval / base
    tabel2 = np.round(const * np.power(base, n))

    ss = np.zeros(l)
    ss[0] = tabel2[0]
    z = np.zeros(l)
    code = np.zeros(l)
    d = np.zeros(l)
    neg = 0
    for n in range(1, l):
        d[n] = x[n] - z[n - 1]
        if d[n] < 0:
            neg = 1
            code[n] += sign_bit
            d[n] = -d[n]
        else:
            neg = 0
        if d[n] >= ss[n - 1]:
            code[n] += 1
        if neg:
            temp = code[n] - sign_bit
        else:
            temp = code[n]
        temp2 = (temp + 0.5) * ss[n - 1]
        if neg:
            temp2 = -temp2
        z[n] = z[n - 1] + temp2
        if z[n] > 127:
            z[n] = 127
        elif z[n] < -127:
            z[n] = -127
        # 计算新的步长
        currentIndex += index[int(temp)]
        if currentIndex < 0:
            currentIndex = 0
        elif currentIndex > numSteps:
            currentIndex = numSteps
        ss[n] = tabel2[currentIndex]
    return code
