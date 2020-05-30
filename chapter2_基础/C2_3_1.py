from soundBase import soundBase
from random import randint
import matplotlib.pyplot as plt
import numpy as np

# 2.2 练习1
sb = soundBase('C2_3_y.wav')
data, fs, nbits = sb.audioread()
# sb.SPL(data, fs)
spl, freq = sb.iso226(50)
