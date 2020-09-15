from scipy.io import loadmat
from chapter2_基础.soundBase import *

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def hide_message(x, meg, nBits=1):
    if nBits != 1:
        exit('Only nBits=1 support now......')
    xx = np.zeros(len(x))
    xx[:] = x[:]
    l = len(meg)
    pads = np.mod(l, nBits)
    if pads:
        l += nBits - pads
        meg_l = np.zeros(l)
        meg_l[:l] = meg
        meg = meg_l
    m_len = l // nBits
    meg_n = meg.reshape(m_len, nBits)
    for i in range(nBits):
        for j in range(m_len):
            if meg_n[j, i]:
                xx[j] = x[j] // 2 * 2
            else:
                xx[j] = x[j] // 2 * 2 + 1
    return xx, m_len


def extract_message(x, m_len, nBits=1):
    if nBits != 1:
        exit('Only nBits=1 support now......')
    meg = np.zeros((m_len, nBits))
    for i in range(nBits):
        for j in range(m_len):
            meg[j, i] = x[j] % 2
    return meg


data, fs, bits = soundBase('C8_1_y.wav').audioread(return_nbits=True)
data16 = (data + 1) * np.power(2, bits - 1)
nBits = 1
s = loadmat('C8_1_y.DAT')

x_embed, m_len = hide_message(data16, s['message'][0], 1)
meg_rec = extract_message(x_embed, m_len, 1)

plt.figure(figsize=(14, 12))
plt.subplot(3, 1, 1)
plt.plot(data16)
plt.subplot(3, 1, 2)
plt.plot(x_embed)
plt.subplot(3, 1, 3)
plt.plot(data16 - x_embed)

plt.show()

plt.subplot(2, 1, 1)
plt.imshow(s['message'][0].reshape(s['n_mess'][0][0], s['m_mess'][0][0]).T)
plt.subplot(2, 1, 2)
plt.imshow(meg_rec.reshape(s['n_mess'][0][0], s['m_mess'][0][0]).T)
plt.show()
