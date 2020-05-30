from chapter2_基础.soundBase import *
from chapter6_语音编码.PCM import *

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

data, fs = soundBase('C6_1_y.wav').audioread()

sxx = np.array(list(map(int, data * 4096)))

y = PCM_encode(sxx)

yy = PCM_decode(y, 1)

plt.subplot(3, 1, 1)
plt.plot(data)
plt.title('编码前')

plt.subplot(3, 1, 2)
plt.plot(yy)
plt.title('解码后')

plt.subplot(3, 1, 3)
plt.plot(yy - data)
plt.title('误差')

plt.savefig('images/pcm.png')
plt.close()
