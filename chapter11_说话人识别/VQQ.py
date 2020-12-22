from chapter2_基础.soundBase import *
from chapter11_说话人识别.VQ import *
from chapter10_语音识别.DTW.DCW import mfccf

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

k = 8
N = 4

## 生成book
u = np.zeros(N)
for i in range(1, N + 1):
    s = 'VQ_data/SX' + str(i) + '.WAV'
    # s = 'VQ_data/mysound.WAV'
    data, fs, bits = soundBase(s).audioread()
    data /= np.max(data)
    mel = mfccf(12, data, fs)
    v = lbg(mel.T, k)
    u[i] = v

## 识别过程
M = 4  # 每个人有M个待识别的样本
l = 5

# 这部分需要用新的语音信号对于MATLAB调试查看结果
for iii in range(l):
    for i in range(M):
        Dstu = np.zeros(N)
        s = 'VQ_data/TX{}_{}.wav'.format(iii, i)
        data, fs, bits = soundBase(s).audioread()
        data /= np.max(data)
        mel = mfccf(12, data, fs)  # 测试数据特征
        for ii in range(N):
            for jj in range(mel.shape[1]):
                distance = dis(u[jj], mel[:, jj])
                Dstu[ii] += np.min(distance)
        pos = np.argmin(Dstu)
        if Dstu[pos] / mel.shape[1] >= 81:
            print('测试者不是系统内人\n')
        else:
            print('测试者为{}'.format(pos + 1))
