from chapter3_分析实验.windows import *
from chapter3_分析实验.timefeature import *
from chapter2_基础.soundBase import *
from chapter3_分析实验.倒谱计算 import *

data, fs = soundBase('C3_4_y_1.wav').audioread()
nlen = 1000
y = data[:nlen]
N = 1024
time = [i / fs for i in range(nlen)]
z = cceps(y)
zr = rcceps(y)
yy = icceps(z)

plt.subplot(4, 1, 1)
plt.plot(time, y)
plt.title('原始信号')
plt.subplot(4, 1, 2)
plt.plot(time, z)
plt.title('复倒谱')
plt.subplot(4, 1, 3)
plt.plot(time, zr)
plt.title('实倒谱')
plt.subplot(4, 1, 4)
plt.plot(time, yy)
plt.title('重构信号')
plt.savefig('images/倒谱.png')
plt.close()
