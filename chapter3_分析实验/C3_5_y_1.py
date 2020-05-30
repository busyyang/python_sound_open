from chapter2_基础.soundBase import *
from chapter3_分析实验.lpc import *
from scipy.signal import lfilter

data, fs = soundBase('C3_5_y.wav').audioread()
L = 240
x = data[8000:8000 + L]
x = (x - np.mean(x)) / np.std(x)
p = 12
ar, G = lpc_coeff(x, p)
b = np.zeros(p + 1)
b[0] = 1
b[1:] = -ar[1:]
est_x = lfilter(b, 1, x)
plt.subplot(2, 1, 1)
plt.plot(x)
plt.subplot(2, 1, 2)
plt.plot(est_x)
plt.savefig('images/lpc.png')
plt.close()
