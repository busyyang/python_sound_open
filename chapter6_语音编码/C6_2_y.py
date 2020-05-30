from scipy.signal import filtfilt

from chapter2_基础.soundBase import *
from chapter3_分析实验.lpc import lpc_coeff

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

data, fs = soundBase('C6_1_y.wav').audioread()
N = len(data)
time = [i / fs for i in range(N)]  # 设置时间
p = 12
ar, g = lpc_coeff(data, p)
ar[0] = 0
est_x = filtfilt(-ar, [1], data)

plt.subplot(2, 1, 1)
plt.plot(time, data, 'k')
plt.plot(time, est_x, 'c')
plt.title('LPC解码')
plt.legend(['信号', '解码信号'])
plt.subplot(2, 1, 2)
plt.plot(est_x - data)
plt.title('误差')
plt.savefig('images/LPC解码.png')
plt.close()
