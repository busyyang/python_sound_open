from chapter2_基础.soundBase import *
from chapter4_特征提取.end_detection import *

data, fs = soundBase('C4_1_y.wav').audioread()
data /= np.max(data)
N = len(data)
wlen = 200
inc = 80
IS = 0.1
overlap = wlen - inc
NIS = int((IS * fs - wlen) // inc + 1)
fn = (N - wlen) // inc + 1

frameTime = FrameTimeC(fn, wlen, inc, fs)
time = [i / fs for i in range(N)]

voiceseg, vsl, SF, NF, amp, zcr = vad_TwoThr(data, wlen, inc, NIS)

plt.subplot(3, 1, 1)
plt.plot(time, data)

plt.subplot(3, 1, 2)
plt.plot(frameTime, amp)

plt.subplot(3, 1, 3)
plt.plot(frameTime, zcr)

for i in range(vsl):
    plt.subplot(3, 1, 1)
    plt.plot(frameTime[voiceseg[i]['start']], 1, '.k')
    plt.plot(frameTime[voiceseg[i]['end']], 1, 'or')

    plt.subplot(3, 1, 2)
    plt.plot(frameTime[voiceseg[i]['start']], 1, '.k')
    plt.plot(frameTime[voiceseg[i]['end']], 1, 'or')

    plt.subplot(3, 1, 3)
    plt.plot(frameTime[voiceseg[i]['start']], 1, '.k')
    plt.plot(frameTime[voiceseg[i]['end']], 1, 'or')

plt.savefig('images/TwoThr.png')
plt.close()
