from chapter2_基础.soundBase import *
from chapter3_分析实验.dct import *

f = 50
fs = 1000
N = 1000
n = np.array([i for i in range(N)])

xn = np.cos(2 * np.pi * f * n / fs)
y = dct(xn)
y = np.where(abs(y) < 5, 0, y)

zn = idct(y)

plt.subplot(3, 1, 1)
plt.plot(xn)
plt.subplot(3, 1, 2)
plt.plot(y)
plt.subplot(3, 1, 3)
plt.plot(zn)
plt.savefig('images/dct.png')
plt.close()
