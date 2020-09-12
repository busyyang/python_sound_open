# encoding: utf-8
'''
@application:
@file: wp_mfcc.py
@time: 2020/9/11 下午 14:39
@desc: 测试小波包-MFCC文件
'''

from chapter2_基础.soundBase import *
from chapter5_语音降噪.Wavelet import *

data, fs, _ = soundBase('C5_4_y.wav').audioread()
data -= np.mean(data)
data /= np.max(np.abs(data))

wpcoeff = wavePacketDec(data, 3, 'db7')
dd = wavePacketRec(wpcoeff, 'db7')

for i in range(len(dd)):
    plt.subplot(len(dd), 2, 2 * i + 1)
    plt.plot(dd[i])
    plt.subplot(len(dd), 2, 2 * i + 2)
    plt.plot(np.linspace(0, fs / 2, len(dd[i]) // 2 + 1), np.abs(np.fft.rfft(dd[i]) / (len(dd[i]) // 2 + 1)) ** 2)
plt.show()

# 使用小波包-MFCC结合提取特征
wmfcc = WPMFCC(data, fs, 12, 200, 100, 3, 'db7')
mfcc = Nmfcc(data, fs, 12, 200, 100)
print(1)
