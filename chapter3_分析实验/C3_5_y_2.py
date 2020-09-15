from chapter2_基础.soundBase import *
from chapter3_分析实验.lpc import *

data, fs = soundBase('C3_5_y.wav').audioread()
L = 240
p = 12
x = data[8000:8000 + L]
ar, G = lpc_coeff(x, p)
nfft = 512
W2 = nfft // 2
m = np.array([i for i in range(W2)])
Y = np.fft.fft(x, nfft)
Y1 = lpcff(ar, W2)
plt.subplot(2, 1, 1)
plt.plot(x)
plt.subplot(2, 1, 2)
plt.plot(m, 20 * np.log(np.abs(Y[m])))
plt.plot(m, 20 * np.log(np.abs(Y1[m])))
plt.savefig('images/lpcff.png')
plt.close()
