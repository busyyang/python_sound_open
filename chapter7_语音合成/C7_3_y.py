from chapter2_基础.soundBase import *
from chapter3_分析实验.timefeature import *
from chapter7_语音合成.flipframe import *
from chapter3_分析实验.C3_1_y_1 import enframe
from chapter3_分析实验.lpc import lpc_coeff
from chapter4_特征提取.共振峰估计 import *

from chapter4_特征提取.pitch_detection import *

from chapter7_语音合成.myfilter import *

from scipy.signal import lfilter

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

data, fs = soundBase('C7_3_y.wav').audioread()
data -= np.mean(data)
data /= np.max(np.abs(data))
data = lfilter([1, -0.99], 1, data)
N = len(data)
time = [i / fs for i in range(N)]  # 设置时间
wlen = 240
inc = 80
overlap = wlen - inc
n2 = [i for i in range(wlen // 2)]
w1 = [i / overlap for i in range(overlap)]
w2 = [i / overlap for i in range(overlap - 1, -1, -1)]
wnd = np.hamming(wlen)
X = enframe(data, wnd, inc)
fn = X.shape[0]
Etmp = np.sum(np.power(X, 2), axis=1)
Etmp /= np.max(Etmp)
T1, r2 = 0.1, 0.5
miniL = 10
mnlong = 5
ThrC = [10, 15]
p = 12

frameTime = FrameTimeC(fn, wlen, inc, fs)
Doption = 0

voiceseg, vosl, SF, Ef, period = pitch_Ceps(data, wlen, inc, T1, fs)
Dpitch = pitfilterm1(period, voiceseg, vosl)
## 共振峰检测
Frmt = np.zeros((3, fn))
Bw = np.zeros((3, fn))
U = np.zeros((3, fn))
for i in range(len(SF)):
    Frmt[:, i], Bw[:, i], U[:, i] = Formant_Root(X[:, ], p, fs, 3)

# 语音合成

zint = np.zeros((2, 4))
tal = 0
for i in range(fn):
    yf = Frmt[:, i]
    bw = Bw[:, i]
## To Be continue

