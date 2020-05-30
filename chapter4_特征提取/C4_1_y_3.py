from chapter2_基础.soundBase import *
from chapter4_特征提取.end_detection import *

data, fs = soundBase('C4_1_y.wav').audioread()
data -= np.mean(data)
data /= np.max(data)
IS = 0.25
wlen = 200
inc = 80
N = len(data)
time = [i / fs for i in range(N)]
wnd = np.hamming(wlen)
overlap = wlen - inc
NIS = int((IS * fs - wlen) // inc + 1)
thr1 = 0.99
thr2 = 0.96
voiceseg, vsl, SF, NF, Enm = vad_specEN(data, wnd, inc, NIS, thr1, thr2, fs)

fn = len(SF)
frameTime = FrameTimeC(fn, wlen, inc, fs)

plt.subplot(2, 1, 1)
plt.plot(time, data)
plt.subplot(2, 1, 2)
plt.plot(frameTime, Enm)

for i in range(vsl):
    plt.subplot(2, 1, 1)
    plt.plot(frameTime[voiceseg[i]['start']], 0, '.k')
    plt.plot(frameTime[voiceseg[i]['end']], 0, 'or')
    plt.legend(['signal', 'start', 'end'])

    plt.subplot(2, 1, 2)
    plt.plot(frameTime[voiceseg[i]['start']], 0, '.k')
    plt.plot(frameTime[voiceseg[i]['end']], 0, 'or')
    plt.legend(['熵谱', 'start', 'end'])

plt.savefig('images/En.png')
plt.close()
