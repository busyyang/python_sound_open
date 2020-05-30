from chapter2_基础.soundBase import *
from chapter5_语音降噪.自适应滤波 import *


def awgn(x, snr):
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)
    npower = xpower / snr
    return x + np.random.randn(len(x)) * np.sqrt(npower)


data, fs = soundBase('C5_1_y.wav').audioread()
data -= np.mean(data)
data /= np.max(np.abs(data))
IS = 0.25  # 设置前导无话段长度
wlen = 200  # 设置帧长为25ms
inc = 80  # 设置帧移为10ms
SNR = 5  # 设置信噪比SNR
N = len(data)  # 信号长度
time = [i / fs for i in range(N)]  # 设置时间
r1 = awgn(data, SNR)
NIS = int((IS * fs - wlen) // inc + 1)
# 5.2.1
snr1 = SNR_Calc(r1, r1 - data)
a, b = 4, 0.001
output = SpectralSub(r1, wlen, inc, NIS, a, b)
if len(output) < len(r1):
    filted = np.zeros(len(r1))
    filted[:len(output)] = output
elif len(output) > len(r1):
    filted = output[:len(r1)]
else:
    filted = output

plt.subplot(4, 1, 1)
plt.plot(time, data)
plt.ylabel('原始信号')
plt.subplot(4, 1, 2)
plt.plot(time, r1)
plt.ylabel('加噪声信号')
plt.subplot(4, 1, 3)
plt.ylabel('滤波信号')
plt.plot(time, filted)

# 5.2.2




plt.show()
