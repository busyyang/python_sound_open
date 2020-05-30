from chapter2_基础.soundBase import *
from chapter3_分析实验.lpc import *
from chapter3_分析实验.倒谱计算 import *

data, fs = soundBase('C3_5_y.wav').audioread()
L = 240
p = 12
x = data[8000:8000 + L]
ar, G = lpc_coeff(x, p)

lpcc1 = lpc_lpccm(ar, p, p)
lpcc2 = rcceps(ar)
plt.subplot(2, 1, 1)
plt.plot(lpcc1)
plt.subplot(2, 1, 2)
plt.plot(lpcc2)
plt.show()
