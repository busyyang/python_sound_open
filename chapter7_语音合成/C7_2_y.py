from chapter2_基础.soundBase import *
from chapter7_语音合成.flipframe import *
from chapter3_分析实验.C3_1_y_1 import enframe
from chapter3_分析实验.lpc import lpc_coeff

from scipy.signal import lfilter

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

data, fs = soundBase('C7_2_y.wav').audioread()

data -= np.mean(data)
data /= np.max(np.abs(data))
N = len(data)
time = [i / fs for i in range(N)]  # 设置时间
p = 12
wlen, inc = 200, 80
msoverlap = wlen - inc
y = enframe(data, wlen, inc)
fn = y.shape[0]
Acoef = np.zeros((y.shape[0], p + 1))
resid = np.zeros(y.shape)
synFrame = np.zeros(y.shape)
## 7.2.1

# 求每帧的LPC系数与预测误差
for i in range(fn):
    a, _ = lpc_coeff(y[i, :], p)
    Acoef[i, :] = a
    resid[i, :] = lfilter(a, [1], y[i, :])

# 语音合成
for i in range(fn):
    synFrame[i, :] = lfilter([1], Acoef[i, :], resid[i, :])

outspeech = Filpframe_OverlapS(synFrame, np.hamming(wlen), inc)
plt.subplot(2, 1, 1)
plt.plot(data / np.max(np.abs(data)), 'k')
plt.title('原始信号')
plt.subplot(2, 1, 2)
plt.title('还原信号-LPC与误差')
plt.plot(outspeech / np.max(np.abs(outspeech)), 'c')

plt.savefig('images/LPC与误差.png')
plt.close()
