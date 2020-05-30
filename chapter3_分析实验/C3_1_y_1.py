from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def enframe(x, win, inc=None):
    nx = len(x)
    if isinstance(win, list) or isinstance(win, np.ndarray):
        nwin = len(win)
        nlen = nwin  # 帧长=窗长
    elif isinstance(win, int):
        nwin = 1
        nlen = win  # 设置为帧长
    if inc is None:
        inc = nlen
    nf = (nx - nlen + inc) // inc
    frameout = np.zeros((nf, nlen))
    indf = np.multiply(inc, np.array([i for i in range(nf)]))
    for i in range(nf):
        frameout[i, :] = x[indf[i]:indf[i] + nlen]
    if isinstance(win, list) or isinstance(win, np.ndarray):
        frameout = np.multiply(frameout, np.array(win))
    return frameout


if __name__ == "__main__":
    fs, data, nbits = wavfile.read('C3_1_y.wav')


    inc = 100
    wlen = 200
    en = enframe(data, wlen, inc)
    i = input('其实帧(i):')
    i = int(i)
    tlabel = i
    plt.subplot(4, 1, 1)
    x = [i for i in range((tlabel - 1) * inc, (tlabel - 1) * inc + wlen)]
    plt.plot(x, en[tlabel, :])
    plt.xlim([(i - 1) * inc + 1, (i + 2) * inc + wlen])
    plt.title('(a)当前波形帧号{}'.format(tlabel))

    plt.subplot(4, 1, 2)
    x = [i for i in range((tlabel + 1 - 1) * inc, (tlabel + 1 - 1) * inc + wlen)]
    plt.plot(x, en[i + 1, :])
    plt.xlim([(i - 1) * inc + 1, (i + 2) * inc + wlen])
    plt.title('(b)当前波形帧号{}'.format(tlabel + 1))

    plt.subplot(4, 1, 3)
    x = [i for i in range((tlabel + 2 - 1) * inc, (tlabel + 2 - 1) * inc + wlen)]
    plt.plot(x, en[i + 2, :])
    plt.xlim([(i - 1) * inc + 1, (i + 2) * inc + wlen])
    plt.title('(c)当前波形帧号{}'.format(tlabel + 2))

    plt.subplot(4, 1, 4)
    x = [i for i in range((tlabel + 3 - 1) * inc, (tlabel + 3 - 1) * inc + wlen)]
    plt.plot(x, en[i + 3, :])
    plt.xlim([(i - 1) * inc + 1, (i + 2) * inc + wlen])
    plt.title('(d)当前波形帧号{}'.format(tlabel + 3))

    # plt.show()
    plt.savefig('images/en.png')
    plt.close()
