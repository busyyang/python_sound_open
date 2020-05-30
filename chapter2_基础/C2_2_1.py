from soundBase import soundBase
from random import randint, random
import matplotlib.pyplot as plt
import numpy as np

# 2.2 练习1
sb = soundBase('C2_2_y.wav')
# 读取语音
'''
这里的wavfile.read()函数修改了里面的代码，返回项return fs, data 改为了return fs, data, bit_depth
如果这里报错，可以将wavfile.read()修改。
:param formater: 获取数据的格式，为sample时，数据为float32的，[-1,1]，同matlab同名函数. 否则为文件本身的数据格式
                指定formater为任意非sample字符串，则返回原始数据。
:return: 语音数据data, 采样率fs，数据位数bits
'''
data, fs, nbits = sb.audioread()
print(fs)
max_data = max(data)
noise = [random() * 0.1 for i in range(len(data))]
fixed2 = sb.sound_add(data, noise)
plt.subplot(311)
plt.plot(data)
plt.subplot(312)
plt.plot(noise)
plt.subplot(313)
plt.plot(fixed2)
plt.show()
sb_f = soundBase('C2_2_y_noised.wav')
sb_f.audiowrite(fixed2, fs)
# sb_f.audioplayer()

# 2.2 练习2
conved = np.convolve(data, noise, 'same')
sb_c = soundBase('C2_2_y_conved.wav')
sb_c.audiowrite(conved, fs)
# sb_c.audioplayer()

# 2.2 练习3
plt.subplot(211)
x = [i / fs for i in range(len(data))]
plt.plot(x, data)
sb_cc = soundBase('C2_2_y_conved_2.wav')
sb_c.audiowrite(data, fs * 2)
'''
这里的wavfile.read()函数修改了里面的代码，返回项return fs, data 改为了return fs, data, bit_depth
如果这里报错，可以将wavfile.read()修改。
:param formater: 获取数据的格式，为sample时，数据为float32的，[-1,1]，同matlab同名函数. 否则为文件本身的数据格式
                指定formater为任意非sample字符串，则返回原始数据。
:return: 语音数据data, 采样率fs，数据位数bits
'''
data, fs_, nbits = sb_c.audioread()
x = [i / fs_ for i in range(len(data))]
print(fs_)
plt.subplot(212)
plt.plot(x, data)
plt.show()
