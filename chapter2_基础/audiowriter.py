import librosa  # 填充，默认频率为22050，可以改变频率
from scipy.io import wavfile  # 原音无损
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

fs, data = wavfile.read('C2_3_y.wav')  # 原始频率，原始数据
print("长度 = {0} 秒".format(len(data) / fs))

data1, sample_rate = librosa.load('C2_3_y.wav', sr=fs)
print("长度 = {0} 秒".format(len(data1) / sample_rate))
plt.plot(data1)
plt.show()

# path = 'C2_1_y_2.wav'
# librosa.output.write_wav(path, data.astype(np.float32), sr=sample_rate)

plt.figure(figsize=(14, 5))
librosa.display.waveplot(data1, sample_rate)
plt.show()
