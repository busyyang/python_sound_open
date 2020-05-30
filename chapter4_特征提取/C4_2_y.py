from chapter2_基础.soundBase import *
from chapter4_特征提取.pitch_detection import *

data, fs = soundBase('C4_2_y.wav').audioread()
data -= np.mean(data)
data /= np.max(np.abs(data))
wlen = 320
inc = 80
N = len(data)
time = [i / fs for i in range(N)]
T1 = 0.05

# 4.2.1
voiceseg, vosl, SF, Ef = pitch_vad(data, wlen, inc, T1)
fn = len(SF)
frameTime = FrameTimeC(fn, wlen, inc, fs)

plt.figure(figsize=(14, 8))

plt.subplot(5, 1, 1)
plt.plot(time, data)
plt.subplot(5, 1, 2)
plt.plot(frameTime, Ef)
for i in range(vosl):
    plt.subplot(5, 1, 2)
    plt.plot(frameTime[voiceseg[i]['start']], Ef[voiceseg[i]['start']], '.k')
    plt.plot(frameTime[voiceseg[i]['end']], Ef[voiceseg[i]['start']], 'or')
    plt.legend(['能熵比', 'start', 'end'])

# 4.2.3
voiceseg, vsl, SF, Ef, period = pitch_Ceps(data, wlen, inc, T1, fs, miniL=10)
plt.subplot(5, 1, 3)
plt.plot(frameTime, period)
for i in range(vsl):
    plt.subplot(5, 1, 3)
    plt.plot(frameTime[voiceseg[i]['start']], Ef[voiceseg[i]['start']], '.k')
    plt.plot(frameTime[voiceseg[i]['end']], Ef[voiceseg[i]['start']], 'or')
    plt.legend(['倒谱法', 'start', 'end'])

# 4.2.4
voiceseg, vsl, SF, Ef, period = pitch_Corr(data, wlen, inc, T1, fs)
plt.subplot(5, 1, 4)
plt.plot(frameTime, period)
for i in range(vsl):
    plt.subplot(5, 1, 4)
    plt.plot(frameTime[voiceseg[i]['start']], Ef[voiceseg[i]['start']], '.k')
    plt.plot(frameTime[voiceseg[i]['end']], Ef[voiceseg[i]['start']], 'or')
    plt.legend(['自相关', 'start', 'end'])

# 4.2.5
p = 12
voiceseg, vsl, SF, Ef, period = pitch_Lpc(data, wlen, inc, T1, fs, p)
plt.subplot(5, 1, 5)
plt.plot(frameTime, period)
for i in range(vsl):
    plt.subplot(5, 1, 5)
    plt.plot(frameTime[voiceseg[i]['start']], Ef[voiceseg[i]['start']], '.k')
    plt.plot(frameTime[voiceseg[i]['end']], Ef[voiceseg[i]['start']], 'or')
    plt.legend(['线性预测', 'start', 'end'])

plt.savefig('images/pitch.png')
plt.close()

# 4.2.2
from scipy.signal import ellipord, ellip, freqz

fs = 8000
fs2 = fs / 2
Wp = np.array([60, 500]) / fs2
Ws = np.array([20, 1500]) / fs2
Rp = 1
Rs = 40
n, Wn = ellipord(Wp, Ws, Rp, Rs)
b, a = ellip(n, Rp, Rs, Wn, 'bandpass')
print(b)
print(a)

w, H = freqz(b, a, 1000)
H, w = H[:500], w[:500]
mag = np.abs(H)
db = 20 * np.log10((mag + 1e-20) / np.max(mag))

plt.plot(w / np.pi * fs2, db)
plt.ylim([-90, 10])
plt.title('椭圆滤波器频率响应')
plt.savefig('images/ellip.png')
