from chapter3_分析实验.mel import Nmfcc
from scipy.io import wavfile, loadmat
from hmmlearn import hmm
from sklearn.externals import joblib
import numpy as np
import os

"""
代码来自：https://blog.csdn.net/chinatelecom08/article/details/82901480
并进行了部分更改
"""


def gen_wavlist(wavpath):
    """
    得到数据文件序列
    :param wavpath:
    :return:
    """
    wavdict = {}
    labeldict = {}
    for (dirpath, dirnames, filenames) in os.walk(wavpath):
        for filename in filenames:
            if filename.endswith('.wav'):
                filepath = os.sep.join([dirpath, filename])
                fileid = filename.strip('.wav')
                wavdict[fileid] = filepath
                label = fileid.split('_')[1]
                labeldict[fileid] = label
    return wavdict, labeldict


def compute_mfcc(file):
    """
    读取数据并计算mfcc
    :param file: 文件名
    :return: mfcc系数
    """
    """
        有手动修改wavfile.read()函数的返回值，添加了bit_depth的返回，如果报错，修改调用方式为：
        fs, audio = wavfile.read(file)
        2020-3-20   Jie Y.
    """
    fs, audio, bits = wavfile.read(file)
    """
        由于部分信号太短而报错，所以fs//2了
    """
    mfcc = Nmfcc(audio, fs // 2, 12, frameSize=int(fs // 2 * 0.025), inc=int(fs // 2 * 0.01))
    return mfcc


'''
&usage:		搭建HMM-GMM的孤立词识别模型
参数意义：
	CATEGORY:	所有标签的列表
	n_comp:		每个孤立词中的状态数
	n_mix:		每个状态包含的混合高斯数量
	cov_type:	协方差矩阵的类型
	n_iter:		训练迭代次数
'''


class Model:
    def __init__(self, CATEGORY=None, n_comp=3, n_mix=3, cov_type='diag', n_iter=1000):
        super(Model, self).__init__()
        self.CATEGORY = CATEGORY
        self.category = len(CATEGORY)
        self.n_comp = n_comp
        self.n_mix = n_mix
        self.cov_type = cov_type
        self.n_iter = n_iter
        # 关键步骤，初始化models，返回特定参数的模型的列表
        self.models = []
        for k in range(self.category):
            model = hmm.GMMHMM(n_components=self.n_comp, n_mix=self.n_mix, covariance_type=self.cov_type,
                               n_iter=self.n_iter)
            self.models.append(model)

    def train(self, tdata):
        for i in range(tdata.shape[1]):
            model = self.models[i]
            for x in range(tdata[0, i].shape[1]):
                data = tdata[0, i][0, x].squeeze()
                mfcc = Nmfcc(data, 8000, 24, 256, 80)
                model.fit(mfcc)

    def test(self, pdata):
        label = []
        result = []
        for k in range(pdata.shape[1]):
            for i in range(pdata[0, k].shape[1]):
                label.append(str(k + 1))
                data = pdata[0, k][0, i].squeeze()
                mfcc = Nmfcc(data, 8000, 24, 256, 80)
                result_one = []
                for m in range(self.category):
                    model = self.models[m]
                    re = model.score(mfcc)
                    result_one.append(re)
                result.append(self.CATEGORY[np.argmax(np.array(result_one))])
        print('识别得到结果：\n', result)
        print('原始标签类别：\n', label)
        # 检查识别率，为：正确识别的个数/总数
        totalnum = len(label)
        correctnum = 0
        for i in range(totalnum):
            if result[i] == label[i]:
                correctnum += 1
        print('识别率:', correctnum / totalnum)

    def save(self, path="models.pkl"):
        joblib.dump(self.models, path)

    def load(self, path="models.pkl"):
        self.models = joblib.load(path)


tdata = loadmat('tra_data.mat')['tdata']
pdata = loadmat('rec_data.mat')['rdata']

CATEGORY = [str(i + 1) for i in range(tdata.shape[1])]
# 进行训练
models = Model(CATEGORY=CATEGORY)
models.train(tdata)
models.test(tdata)
models.test(pdata)
