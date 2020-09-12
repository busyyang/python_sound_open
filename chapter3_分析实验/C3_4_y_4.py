from chapter2_基础.soundBase import *
from chapter3_分析实验.dct import *
from chapter3_分析实验.mel import *

data, fs, _ = soundBase('C3_4_y_4.wav').audioread()

wlen = 200
inc = 80
num = 8
data = data / np.max(data)
mfcc = Nmfcc(data, fs, num, wlen, inc)
