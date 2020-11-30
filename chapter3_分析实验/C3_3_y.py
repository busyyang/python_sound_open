from chapter3_分析实验.windows import *
from chapter3_分析实验.timefeature import *
from chapter2_基础.soundBase import *


def STFFT(x, win, nfft, inc):
    xn = enframe(x, win, inc)
    xn = xn.T
    y = np.fft.fft(xn, nfft, axis=0)
    return y[:nfft // 2, :]


data, fs = soundBase('C3_3_y.wav').audioread()

wlen = 256
nfft = wlen
win = hanning_window(wlen)
inc = 128

y = STFFT(data, win, nfft, inc)
freq = [i * fs / wlen for i in range(wlen // 2)]
frame = FrameTimeC(y.shape[1], wlen, inc, fs)

plt.matshow(np.log10(np.flip(np.abs(y) * np.abs(y), 0)))
plt.colorbar()
plt.savefig('images/spec.png')
plt.close()

plt.specgram(data, NFFT=256, Fs=fs, window=np.hanning(256))
plt.ylabel('Frequency')
plt.xlabel('Time(s)')
plt.show()
