from chapter2_基础.soundBase import *
from chapter7_语音合成.flipframe import *
from chapter3_分析实验.C3_1_y_1 import enframe

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

data, fs = soundBase('C7_1_y.wav').audioread()

wlen = 256
wnd = np.hamming(wlen)
overlap = 100
f = enframe(data, wnd, overlap)
plt.figure(figsize=(14, 12))
# 7.1.1
fn_overlap = Filpframe_OverlapA(f, wnd, overlap)
plt.subplot(3, 2, 1)
plt.plot(data / np.max(np.abs(data)), 'k')
plt.title('原始信号')
plt.subplot(3, 2, 2)
plt.title('还原信号-重叠相加法')
plt.plot(fn_overlap / np.max(np.abs(fn_overlap)), 'c')

# 7.1.2
fn_s = Filpframe_OverlapS(f, wnd, overlap)
plt.subplot(3, 2, 3)
plt.plot(data / np.max(np.abs(data)), 'k')
plt.title('原始信号')
plt.subplot(3, 2, 4)
plt.title('还原信号-重叠存储法')
plt.plot(fn_s / np.max(np.abs(fn_s)), 'c')

# 7.1.3
fn_l = Filpframe_LinearA(f, wnd, overlap)
plt.subplot(3, 2, 5)
plt.plot(data / np.max(np.abs(data)), 'k')
plt.title('原始信号')
plt.subplot(3, 2, 6)
plt.title('还原信号-线性叠加法')
plt.plot(fn_l / np.max(np.abs(fn_l)), 'c')

plt.savefig('images/flipframe.png')
plt.close()
