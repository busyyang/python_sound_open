import numpy as np


def PCM_encode(x):
    n = len(x)
    out = np.zeros((n, 8))
    for i in range(n):
        # 符号位
        if x[i] > 0:
            out[i, 0] = 1
        else:
            out[i, 0] = 0
        # 数据位
        if abs(x[i]) < 32:
            out[i, 1], out[i, 2], out[i, 3], step, st = 0, 0, 0, 2, 0
        elif abs(x[i]) < 64:
            out[i, 1], out[i, 2], out[i, 3], step, st = 0, 0, 1, 2, 32
        elif abs(x[i]) < 128:
            out[i, 1], out[i, 2], out[i, 3], step, st = 0, 1, 0, 4, 64
        elif abs(x[i]) < 256:
            out[i, 1], out[i, 2], out[i, 3], step, st = 0, 1, 1, 8, 128
        elif abs(x[i]) < 512:
            out[i, 1], out[i, 2], out[i, 3], step, st = 1, 0, 0, 16, 256
        elif abs(x[i]) < 1024:
            out[i, 1], out[i, 2], out[i, 3], step, st = 1, 0, 1, 32, 512
        elif abs(x[i]) < 2048:
            out[i, 1], out[i, 2], out[i, 3], step, st = 1, 1, 0, 64, 1024
        else:
            out[i, 1], out[i, 2], out[i, 3], step, st = 1, 1, 1, 128, 2048

        if abs(x[i]) >= 4096:
            out[i, 1:] = np.array([1, 1, 1, 1, 1, 1])
        else:
            tmp = bin(int((abs(x[i]) - st) / step)).replace('0b', '')
            tmp = '0' * (4 - len(tmp)) + tmp
            t = [int(k) for k in tmp]
            out[i, 4:] = t
    return out.reshape(8 * n)


def PCM_decode(ins, v):
    inn = ins.reshape(len(ins) // 8, 8)
    slot = np.array([0, 32, 64, 128, 256, 512, 1024, 2048])
    step = np.array([2, 2, 4, 8, 16, 32, 64, 128])
    out = np.zeros(len(ins) // 8)
    for i in range(inn.shape[0]):
        sgn = 2 * inn[i, 0] - 1
        tmp = int(inn[i, 1] * 4 + inn[i, 2] * 2 + inn[i, 3])
        st = slot[tmp]
        dt = (inn[i, 4] * 8 + inn[i, 5] * 4 + inn[i, 6] * 2 + inn[i, 7]) * step[tmp] + 0.5 * step[tmp]
        out[i] = sgn * (st + dt) / 4096 * v
    return out
