from chapter2_基础.soundBase import *
from chapter4_特征提取.end_detection import *


def awgn(x, snr):
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower) + x


data, fs = soundBase('C4_1_y.wav').audioread()
data -= np.mean(data)
data /= np.max(data)
IS = 0.25
wlen = 200
inc = 80
SNR = 10
N = len(data)
time = [i / fs for i in range(N)]
wnd = np.hamming(wlen)
overlap = wlen - inc
NIS = int((IS * fs - wlen) // inc + 1)
signal = awgn(data, SNR)

y = enframe(signal, wnd, inc)
frameTime = FrameTimeC(y.shape[0], wlen, inc, fs)

Y = np.abs(np.fft.fft(y, axis=1))
Y = Y[:, :wlen // 2]
N = np.mean(Y[:NIS, :], axis=0)
NoiseCounter = 0
SF = np.zeros(y.shape[0])
NF = np.zeros(y.shape[0])
D = np.zeros(y.shape[0])
# 前导段设置NF=1,SF=0
SF[:NIS] = 0
NF[:NIS] = 1
for i in range(NIS, y.shape[0]):
    NoiseFlag, SpeechFlag, NoiseCounter, Dist = vad_LogSpec(Y[i, :], N, NoiseCounter, 2.5, 8)
    SF[i] = SpeechFlag
    NF[i] = NoiseFlag
    D[i] = Dist
sindex = np.where(SF == 1)
voiceseg = findSegment(np.where(SF == 1)[0])
vosl = len(voiceseg)

plt.subplot(3, 1, 1)
plt.plot(time, data)
plt.subplot(3, 1, 2)
plt.plot(time, signal)
plt.subplot(3, 1, 3)
plt.plot(frameTime, D)

for i in range(vosl):
    plt.subplot(3, 1, 1)
    plt.plot(frameTime[voiceseg[i]['start']], 0, '.k')
    plt.plot(frameTime[voiceseg[i]['end']], 0, 'or')
    plt.legend(['signal', 'start', 'end'])

    plt.subplot(3, 1, 2)
    plt.plot(frameTime[voiceseg[i]['start']], 0, '.k')
    plt.plot(frameTime[voiceseg[i]['end']], 0, 'or')
    plt.legend(['noised', 'start', 'end'])

    plt.subplot(3, 1, 3)
    plt.plot(frameTime[voiceseg[i]['start']], 1, '.k')
    plt.plot(frameTime[voiceseg[i]['end']], 1, 'or')
    plt.legend(['对数频率距离', 'start', 'end'])

plt.savefig('images/对数频率距离.png')
plt.close()
