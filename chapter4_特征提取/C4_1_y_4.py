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

mode = 2
if mode == 1:
    thr1 = 3
    thr2 = 4
    tlabel = '能零比'
elif mode == 2:
    thr1 = 0.05
    thr2 = 0.1
    tlabel = '能熵比'
voiceseg, vsl, SF, NF, Epara = vad_pro(data, wnd, inc, NIS, thr1, thr2, mode)

fn = len(SF)
frameTime = FrameTimeC(fn, wlen, inc, fs)

plt.subplot(2, 1, 1)
plt.plot(time, data)
plt.subplot(2, 1, 2)
plt.plot(frameTime, Epara)

for i in range(vsl):
    plt.subplot(2, 1, 1)
    plt.plot(frameTime[voiceseg[i]['start']], 0, '.k')
    plt.plot(frameTime[voiceseg[i]['end']], 0, 'or')
    plt.legend(['signal', 'start', 'end'])

    plt.subplot(2, 1, 2)
    plt.plot(frameTime[voiceseg[i]['start']], 0, '.k')
    plt.plot(frameTime[voiceseg[i]['end']], 0, 'or')
    plt.legend([tlabel, 'start', 'end'])

plt.savefig('images/{}.png'.format(tlabel))
plt.close()
