from chapter2_基础.soundBase import *
from chapter6_语音编码.ADPCM import *

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

data, fs = soundBase('C6_3_y.wav').audioread()
N = len(data)
time = [i / fs for i in range(N)]  # 设置时间
sig_bit = 2

ss = adpcm_encoder(data, sig_bit)
yy = adpcm_decoder(ss, sig_bit)

plt.subplot(2, 1, 1)
plt.plot(time, data / np.max(data), 'k')
plt.plot(time, yy / np.max(yy), 'c')
plt.title('ADPCM解码')
plt.legend(['信号', '解码信号'])
plt.subplot(2, 1, 2)
plt.plot(data / np.max(data) - yy / np.max(yy))
plt.title('误差')
plt.savefig('images/ADPMC.png')
plt.close()
