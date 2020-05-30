import numpy as np
import random


class HMM:
    def __init__(self, N, M):
        self.N = N
        self.M = M
        self.init = np.zeros(N)
        self.init[0] = 1
        self.trans = np.zeros((N, N))


def kmeans1(d, k):
    n, p = d.shape
    # 任取k个作为初始化
    per = np.random.permutation(n)[:k]
    x = d[per, :]
    y = x + 1


def InitHMM(obs, N, M):
    def getmix(vector, M):
        mean, esp, nn = kmeans1(vector, M)

    print(1)
    K = len(obs.keys())  # 樣本數量
    hmm = HMM(N, M)
    # 初始化狀態轉移矩陣A
    for i in range(N - 1):
        hmm.trans[i, i] = 0.5
        hmm.trans[i, i + 1] = 0.5
    hmm.trans[N - 1, N - 1] = 1
    # 初始化输出观测值概率B（连续混合正态分布）
    for k in range(K):
        T = obs[k]['fea'].shape[0]
        seg = np.linspace(0, T, N + 1)
        obs[k]['segment'] = np.array([int(i) for i in seg])

    for i in range(N):
        for k in range(K):
            seg1 = obs[k]['segment'][i]
            seg2 = obs[k]['segment'][i + 1]
            if k == 0:
                vector = obs[k]['fea'][seg1:seg2, :]
            else:
                vector = np.vstack((vector, obs[k]['fea'][seg1:seg2, :]))
        mix = getmix(vector, M[i])
    return hmm
