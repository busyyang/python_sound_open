from chapter2_基础.soundBase import *
from chapter11_说话人识别.GMM import *
from scipy.io import loadmat
from chapter3_分析实验.C3_1_y_1 import enframe
from chapter3_分析实验.mel import melbankm
from sklearn.mixture import GMM

from chapter10_语音识别.DTW.DTW import mfccf
import warnings

warnings.filterwarnings("ignore")


def Nmfcc(x, fs, p, frameSize, inc):
    """
    计算mfcc系数
    :param x: 输入信号
    :param fs: 采样率
    :param p: Mel滤波器组的个数
    :param frameSize: 分帧的每帧长度
    :param inc: 帧移
    :return:
    """
    # 预处理-预加重
    xx = lfilter([1, -0.97], [1], x)
    # 预处理-分幀
    xx = enframe(xx, frameSize, inc)
    # 预处理-加窗
    xx = np.multiply(xx, np.hanning(frameSize))
    # 计算FFT
    xx = np.fft.fft(xx)
    # 计算能量谱
    xx = np.multiply(np.abs(xx), np.abs(xx))
    # 计算通过Mel滤波器的能量
    xx = xx[:, :frameSize // 2 + 1]
    bank = melbankm(p, frameSize, fs, 0, 0.5 * fs, 0)
    ss = np.matmul(xx, bank.T)
    # 计算DCT倒谱
    n_dct = 20
    M = bank.shape[0]
    m = np.array([i for i in range(M)])
    mfcc = np.zeros((ss.shape[0], n_dct))
    for n in range(n_dct):
        mfcc[:, n] = np.sqrt(2 / M) * np.sum(np.multiply(np.log(ss), np.cos((2 * m - 1) * n * np.pi / 2 / M)), axis=1)
    return mfcc


def train():
    tdata = loadmat('gmm_data/tra_data.mat')['tdata']
    models = {}
    for spk_cyc in range(Spk_num):
        print('训练第{}个说话人'.format(spk_cyc))
        ## 提取MFCC特征
        gmm = GMM(n_components=16, n_iter=200, covariance_type='diag', n_init=3)
        for sp in range(Tra_num):
            speech = tdata[0, spk_cyc][0, sp].squeeze()
            # mfcc = Nmfcc(speech, fs, 20, int(fs * 0.02), int(fs * 0.01))
            mfcc = mfccf(12, speech, fs)
            cc = mfcc[:, :-1]
            ## 训练
            # kiter = 5  # Kmeans的最大迭代次数
            # emiter = 30  # EM算法的最大迭代次数
            # mix = gmm_init(ncentres, cof.T, emiter, 'full')
            # mix, post, errlog = gmm_em(mix, cof.T, emiter)
            gmm.fit(cc)
        models[spk_cyc] = gmm
    return models



def test(models):
    rdata = loadmat('gmm_data/rec_data.mat')['rdata']
    all = 0
    tp = 0
    for i in range(rdata.shape[1]):
        for j in range(rdata[0, i].shape[1]):
            data = rdata[0, i][0, j].squeeze()
            # mfcc = Nmfcc(data, fs, 20, int(fs * 0.02), int(fs * 0.01))
            mfcc = mfccf(12, data, fs)
            cc = mfcc[:, :-1]
            log_likelihood = np.zeros(len(models.values()))
            for key in models.keys():
                score = models[key].score(cc)
                log_likelihood[key] = np.sum(score)
            r = np.argmax(log_likelihood)
            print('第{i}个人的第{j}条语音，被识别为{r}'.format(i=i, j=j, r=r))
            all += 1
            if r == i:
                tp += 1
    print('acc={}'.format(tp / all))


if __name__ == '__main__':
    Spk_num = 6  # 说话人数
    Tra_num = 5  # 每个说话人的训练数据条数
    ncentres = 16  # 混合成分数量
    fs = 16000
    models = train()
    test(models)
