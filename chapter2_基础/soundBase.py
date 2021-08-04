import pyaudio
import wave
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import pandas as pd
from scipy.signal import lfilter


class soundBase:
    def __init__(self, path):
        self.path = path

    def audiorecorder(self, len=2, formater=pyaudio.paInt16, rate=16000, frames_per_buffer=1024, channels=2):
        """
        使用麦克风进行录音
        2020-2-25   Jie Y.  Init
        :param len: 录制时间长度(秒)
        :param formater: 格式
        :param rate: 采样率
        :param frames_per_buffer:
        :param channels: 通道数
        :return:
        """
        p = pyaudio.PyAudio()
        stream = p.open(format=formater, channels=channels, rate=rate, input=True, frames_per_buffer=frames_per_buffer)
        print("start recording......")
        frames = []
        for i in range(0, int(rate / frames_per_buffer * len)):
            data = stream.read(frames_per_buffer)
            frames.append(data)
        print("stop recording......")
        stream.stop_stream()
        stream.close()
        p.terminate()
        wf = wave.open(self.path, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(formater))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
        wf.close()

    def audioplayer(self, frames_per_buffer=1024):
        """
        播放语音文件
        2020-2-25   Jie Y.  Init
        :param frames_per_buffer:
        :return:
        """
        wf = wave.open(self.path, 'rb')
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)
        data = wf.readframes(frames_per_buffer)
        while data != b'':
            stream.write(data)
            data = wf.readframes(frames_per_buffer)

        stream.stop_stream()
        stream.close()
        p.terminate()

    def audiowrite(self, data, fs, binary=True, channel=1, path=[]):
        """
        信息写入到.wav文件中
        :param data: 语音信息数据
        :param fs: 采样率(Hz)
        :param binary: 是否写成二进制文件(只有在写成二进制文件才能用audioplayer播放)
        :param channel: 通道数
        :param path: 文件路径，默认为self.path的路径
        :return:
        """
        if len(path) == 0:
            path = self.path
        if binary:
            wf = wave.open(path, 'wb')
            wf.setframerate(fs)
            wf.setnchannels(channel)
            wf.setsampwidth(2)
            wf.writeframes(b''.join(data))
        else:
            wavfile.write(path, fs, data)

    def audioread(self, return_nbits=False, formater='sample'):
        """
        读取语音文件
        2020-2-26   Jie Y.  Init
        这里的wavfile.read()函数修改了里面的代码，返回项return fs, data 改为了return fs, data, bit_depth
        如果这里报错，可以将wavfile.read()修改。
        :param formater: 获取数据的格式，为sample时，数据为float32的，[-1,1]，同matlab同名函数. 否则为文件本身的数据格式
                        指定formater为任意非sample字符串，则返回原始数据。
        :return: 语音数据data, 采样率fs，数据位数bits
        """
        fs, data, bits = wavfile.read(self.path)
        if formater == 'sample':
            data = data / (2 ** (bits - 1))
        if return_nbits:
            return data, fs, bits
        else:
            return data, fs

    def soundplot(self, data=[], sr=16000, size=(14, 5)):
        """
        将语音数据/或读取语音数据并绘制出来
        2020-2-25   Jie Y.  Init
        :param data: 语音数据
        :param sr: 采样率
        :param size: 绘图窗口大小
        :return:
        """
        if len(data) == 0:
            data, fs, _ = self.audioread()
        plt.figure(figsize=size)
        x = [i / sr for i in range(len(data))]
        plt.plot(x, data)
        plt.xlim([0, len(data) / sr])
        plt.xlabel('s')
        plt.show()

    def sound_add(self, data1, data2):
        """
        将两个信号序列相加，若长短不一，在短的序列后端补零
        :param data1: 序列1
        :param data2: 序列2
        :return:
        """
        if len(data1) < len(data2):
            tmp = np.zeros([len(data2)])
            for i in range(len(data1)):
                tmp[i] += data1[i]
            return tmp + data2
        elif len(data1) > len(data2):
            tmp = np.zeros([len(data1)])
            for i in range(len(data2)):
                tmp[i] += data2[i]
            return tmp + data1
        else:
            return data1 + data2

    def SPL(self, data, fs, frameLen=100, isplot=True):
        """
        计算声压曲线
        2020-2-26   Jie Y.  Init
        :param data: 语音信号数据
        :param fs: 采样率
        :param frameLen: 计算声压的时间长度(ms单位)
        :param isplot: 是否绘图，默认是
        :return: 返回声压列表spls
        """

        def spl_cal(s, fs, frameLen):
            """
            根据数学公式计算单个声压值
            $y=\sqrt(\sum_{i=1}^Nx^2(i))$
            2020-2-26   Jie Y. Init
            :param s: 输入数据
            :param fs: 采样率
            :param frameLen: 计算声压的时间长度(ms单位)
            :return: 单个声压数值
            """
            l = len(s)
            M = frameLen * fs / 1000
            if not l == M:
                exit('输入信号长度与所定义帧长不等！')
            # 计算有效声压
            pp = 0
            for i in range(int(M)):
                pp += (s[i] * s[i])
            pa = np.sqrt(pp / M)
            p0 = 2e-5
            spl = 20 * np.log10(pa / p0)
            return spl

        length = len(data)
        M = fs * frameLen // 1000
        m = length % M
        if not m < M // 2:
            # 最后一帧长度不小于M的一半
            data = np.hstack((data, np.zeros(M - m)))
        else:
            # 最后一帧长度小于M的一半
            data = data[:M * (length // M)]
        spls = np.zeros(len(data) // M)
        for i in range(len(data)// M - 1):
            s = data[i * M:(i + 1) * M]
            spls[i] = spl_cal(s, fs, frameLen)

        if isplot:
            plt.subplot(211)
            plt.plot(data)
            plt.subplot(212)
            plt.step([i for i in range(len(spls))], spls)
            plt.show()
        return spls

    def iso226(self, phon, isplot=True):
        """
        绘制等响度曲线，输入响度phon
        2020-2-26   Jie Y.  Init
        :param phon: 响度值0~90
        :param isplot: 是否绘图，默认是
        :return:
        """
        ## 参数来源: 语音信号处理试验教程，梁瑞宇P36-P37
        f = [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, \
             1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500]
        af = [0.532, 0.506, 0.480, 0.455, 0.432, 0.409, 0.387, 0.367, 0.349, 0.330, 0.315, \
              0.301, 0.288, 0.276, 0.267, 0.259, 0.253, 0.250, 0.246, 0.244, 0.243, 0.243, \
              0.243, 0.242, 0.242, 0.245, 0.254, 0.271, 0.301]

        Lu = [-31.6, - 27.2, - 23.0, - 19.1, - 15.9, - 13.0, - 10.3, - 8.1, - 6.2, - 4.5, - 3.1, \
              - 2.0, - 1.1, - 0.4, 0.0, 0.3, 0.5, 0.0, - 2.7, - 4.1, - 1.0, 1.7, \
              2.5, 1.2, - 2.1, - 7.1, - 11.2, - 10.7, - 3.1]

        Tf = [78.5, 68.7, 59.5, 51.1, 44.0, 37.5, 31.5, 26.5, 22.1, 17.9, 14.4, \
              11.4, 8.6, 6.2, 4.4, 3.0, 2.2, 2.4, 3.5, 1.7, - 1.3, - 4.2, \
              - 6.0, - 5.4, - 1.5, 6.0, 12.6, 13.9, 12.3]
        if phon < 0 or phon > 90:
            print('Phon value out of range!')
            spl = 0
            freq = 0
        else:
            Ln = phon
            # 从响度级计算声压级
            Af = 4.47E-3 * (10 ** (0.025 * Ln) - 1.15) + np.power(0.4 * np.power(10, np.add(Tf, Lu) / 10 - 9), af)
            Lp = np.multiply(np.divide(10, af), np.log10(Af)) - Lu + 94
            spl = Lp
            freq = f
            if isplot:
                plt.semilogx(freq, spl, ':k')
                plt.axis([20, 20000, -10, 130])
                plt.title('Phon={}'.format(phon))
                plt.grid()
                plt.show()
        return spl, freq

    def vowel_generate(self, len, pitch=100, sr=16000, f=[730, 1090, 2440]):
        """
        生成一个元音片段
        2020-2-26   Jie Y.  Init
        :param len: 长度，点数
        :param pitch:
        :param sr: 采样率
        :param f: 前3个共振峰，默认为元音a的
        :return: 生成的序列
        """
        f1, f2, f3 = f[0], f[1], f[2]
        y = np.zeros(len)
        points = [i for i in range(0, len, sr // pitch)]
        indices = np.array(list(map(int, np.floor(points))))
        y[indices] = (indices + 1) - points
        y[indices + 1] = points - indices

        a = np.exp(-250 * 2 * np.pi / sr)
        y = lfilter([1], [1, 0, -a * a], y)
        if f1 > 0:
            cft = f1 / sr
            bw = 50
            q = f1 / bw
            rho = np.exp(-np.pi * cft / q)
            theta = 2 * np.pi * cft * np.sqrt(1 - 1 / (4 * q * q))
            a2 = -2 * rho * np.cos(theta)
            a3 = rho * rho
            y = lfilter([1 + a2 + a3], [1, a2, a3], y)

        if f2 > 0:
            cft = f2 / sr
            bw = 50
            q = f2 / bw
            rho = np.exp(-np.pi * cft / q)
            theta = 2 * np.pi * cft * np.sqrt(1 - 1 / (4 * q * q))
            a2 = -2 * rho * np.cos(theta)
            a3 = rho * rho
            y = lfilter([1 + a2 + a3], [1, a2, a3], y)
        if f3 > 0:
            cft = f3 / sr
            bw = 50
            q = f3 / bw
            rho = np.exp(-np.pi * cft / q)
            theta = 2 * np.pi * cft * np.sqrt(1 - 1 / (4 * q * q))
            a2 = -2 * rho * np.cos(theta)
            a3 = rho * rho
            y = lfilter([1 + a2 + a3], [1, a2, a3], y)
        plt.plot(y)
        plt.show()
        return y
