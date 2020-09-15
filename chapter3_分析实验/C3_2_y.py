from scipy.io import wavfile
import matplotlib.pyplot as plt
from chapter3_分析实验.windows import *
from chapter3_分析实验.timefeature import *
from chapter2_基础.soundBase import *

data, fs = soundBase('C3_2_y.wav').audioread()



inc = 100
wlen = 200
win = hanning_window(wlen)
N = len(data)
time = [i / fs for i in range(N)]

EN = STEn(data, win, inc)  # 短时能量
Mn = STMn(data, win, inc)  # 短时平均幅度
Zcr = STZcr(data, win, inc)  # 短时过零率

X = enframe(data, win, inc)
X = X.T
Ac = STAc(X)
Ac = Ac.T
Ac = Ac.flatten()

Amdf = STAmdf(X)
Amdf = Amdf.flatten()

fig = plt.figure(figsize=(14, 13))
plt.subplot(3, 1, 1)
plt.plot(time, data)
plt.title('(a)语音波形')
plt.subplot(3, 1, 2)
frameTime = FrameTimeC(len(EN), wlen, inc, fs)
plt.plot(frameTime, Mn)
plt.title('(b)短时幅值')
plt.subplot(3, 1, 3)
plt.plot(frameTime, EN)
plt.title('(c)短时能量')
plt.savefig('images/energy.png')

fig = plt.figure(figsize=(10, 13))
plt.subplot(2, 1, 1)
plt.plot(time, data)
plt.title('(a)语音波形')
plt.subplot(2, 1, 2)
plt.plot(frameTime, Zcr)
plt.title('(b)短时过零率')
plt.savefig('images/Zcr.png')

fig = plt.figure(figsize=(10, 13))
plt.subplot(2, 1, 1)
plt.plot(time, data)
plt.title('(a)语音波形')
plt.subplot(2, 1, 2)
plt.plot(Ac)
plt.title('(b)短时自相关')
plt.savefig('images/corr.png')

fig = plt.figure(figsize=(10, 13))
plt.subplot(2, 1, 1)
plt.plot(time, data)
plt.title('(a)语音波形')
plt.subplot(2, 1, 2)
plt.plot(Amdf)
plt.title('(b)短时幅度差')
plt.savefig('images/Amdf.png')
