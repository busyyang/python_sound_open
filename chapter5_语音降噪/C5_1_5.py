from chapter2_基础.soundBase import *
from chapter4_特征提取.共振峰估计 import *
from chapter5_语音降噪.自适应滤波 import *


def awgn(x, snr):
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)
    npower = xpower / snr
    return x + np.random.randn(len(x)) * np.sqrt(npower)


data, fs = soundBase('C5_1_y.wav').audioread()
data -= np.mean(data)
data /= np.max(np.abs(data))

# 5.1.1
N = len(data)
time = [i / fs for i in range(N)]
SNR = 5

r1 = awgn(data, SNR)
M, mu = 64, 0.001
itr = len(r1)
snr1 = SNR_Calc(data, r1 - data)
[y, W, e] = LMS(r1, data, M, mu, itr)
[yn, Wn, en] = NLMS(r1, data, M, mu, itr)
output = e / np.max(np.abs(e))
outputn = en / np.max(np.abs(en))
snr2 = SNR_Calc(data, data - output)
snr2n = SNR_Calc(data, data - outputn)
plt.subplot(4, 1, 1)
plt.plot(time, data)
plt.ylabel('原始信号')
plt.subplot(4, 1, 2)
plt.ylabel('加入噪声')
plt.plot(time, r1)
plt.subplot(4, 1, 3)
plt.ylabel('LMS去噪')
plt.plot(time, output)

plt.subplot(4, 1, 4)
plt.ylabel('NLMS去噪')
plt.plot(time, outputn)
plt.savefig('images/LMS.png')
plt.close()

print('加入噪声SNR:{:.4f}\tLMS滤波后噪声SNR:{:.4f}\t下降SNR:{:.4f}'.format(snr1, snr2, snr2 - snr1))
print('加入噪声SNR:{:.4f}\tNLMS滤波后噪声SNR:{:.4f}\t下降SNR:{:.4f}'.format(snr1, snr2n, snr2n - snr1))
