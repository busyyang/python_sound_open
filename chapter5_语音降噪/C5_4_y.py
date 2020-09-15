from chapter2_基础.soundBase import *
from chapter5_语音降噪.Wavelet import *

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def awgn(x, snr):
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)
    npower = xpower / snr
    return x + np.random.randn(len(x)) * np.sqrt(npower)


data, fs = soundBase('C5_4_y.wav').audioread()
data -= np.mean(data)
data /= np.max(np.abs(data))
SNR = 5
N = len(data)
s = awgn(data, SNR)
time = [i / fs for i in range(N)]  # 设置时间

wname = 'db7'
jN = 6

res_s = Wavelet_Soft(s, jN, wname)
res_d = Wavelet_Hard(s, jN, wname)
res_hs = Wavelet_hardSoft(s, jN, wname)
res_a = Wavelet_average(s, jN, wname)

plt.figure(figsize=(14, 10))
plt.subplot(3, 2, 1)
plt.plot(time, data)
plt.ylabel('原始信号')
plt.subplot(3, 2, 2)
plt.plot(time, s)
plt.ylabel('加噪声信号')
plt.subplot(3, 2, 3)
plt.ylabel('小波软阈值滤波')
plt.plot(time, res_s)

plt.subplot(3, 2, 4)
plt.ylabel('小波硬阈值滤波')
plt.plot(time, res_d)

plt.subplot(3, 2, 5)
plt.ylabel('小波折中阈值滤波')
plt.plot(time, res_hs)

plt.subplot(3, 2, 6)
plt.ylabel('小波加权滤波')
plt.plot(time, res_a)

plt.savefig('images/wavelet.png')
plt.close()
