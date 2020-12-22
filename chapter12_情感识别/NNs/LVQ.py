import numpy as np
import random
from scipy.io import loadmat
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt


def confusion_matrix_info(y_true, y_pred, labels=['fear', 'happy', 'neutr', 'sad', 'anger'],
                          title='confusion matrix'):
    """
    计算混淆矩阵以及一些评价指标，并将混淆矩阵绘图出来
    :param y_true: 真实标签，非one-hot编码
    :param y_pred: 预测标签，非one-hot编码
    :param labels: 标签的含义
    :param title: 绘图的标题
    :return:
    """
    import seaborn as sns
    import pandas as pd
    C2 = confusion_matrix(y_true, y_pred)
    C = pd.DataFrame(C2, columns=labels, index=labels)
    m, _ = C2.shape
    for i in range(m):
        precision = C2[i, i] / sum(C2[:, i])
        recall = C2[i, i] / sum(C2[i, :])
        f1 = 2 * precision * recall / (precision + recall)
        print('In class {}:\t total samples: {}\t true predict samples: {}\t'
              'acc={:.4f},\trecall={:.4f},\tf1-score={:.4f}'.format(
            labels[i], sum(C2[i, :]), C2[i, i], precision, recall, f1))
    print('-' * 100, '\n', 'average f1={:.4f}'.format(f1_score(y_true, y_pred, average='micro')))

    f, ax = plt.subplots()
    sns.heatmap(C, annot=True, ax=ax, cmap=plt.cm.binary)
    ax.set_title(title)
    ax.set_xlabel('predict')
    ax.set_ylabel('true')
    plt.show()


class LVQnet:
    def __init__(self, input_sz, output_sz, groups):
        '''
        初始化，给出输入向量的维度和输出的种类数
        groups是竞争层的分组状况，如[1,2,3,2]
        意为竞争层共有8个神经元，4组输出
        '''
        assert len(groups) == output_sz
        self.groups = groups
        self.hidden_sz = sum(groups)
        # 随机初始化神经元的原型向量
        self.prototype = np.random.rand(self.hidden_sz, input_sz) * 0.01
        self.hidden2out = np.zeros((output_sz, self.hidden_sz))
        cnt = 0
        for i in range(len(groups)):
            for j in range(groups[i]):
                self.hidden2out[i][cnt] = 1
                cnt += 1

    def fit(self, X, Y, lr=0.5, iterations=10000):
        N = len(X)
        for t in range(iterations):
            gamma = lr * (1 - t / iterations)
            idx = random.randint(0, N - 1)
            x = X[idx]
            out = self.predict(x)
            y = Y[idx]
            delta = abs(out - y)
            sign = int(np.sum(delta) == 0) * 2 - 1
            # 根据delta修正获胜神经元的原型向量
            self.prototype[self.winner] += gamma * sign * self.v[self.winner]

    def predict_mat(self, x):
        l = x.shape[0]
        result = []
        for i in range(l):
            out = self.predict(x[i, :])
            result.append(out)
        return np.array(result)

    def predict(self, x):
        x = np.tile(x, (self.hidden_sz, 1))
        v = x - self.prototype
        self.v = v
        distance = np.sum(v ** 2, axis=1).reshape(-1)
        winner = np.argmin(distance)
        self.winner = winner
        out = np.zeros((self.hidden_sz, 1))
        out[winner][0] = 1
        out = self.hidden2out.dot(out).reshape(-1)
        return out


fear = loadmat('A_Fear.mat')['fearVec']
happy = loadmat('F_Happiness.mat')['hapVec']
neutral = loadmat('N_neutral.mat')['neutralVec']
sadness = loadmat('T_sadness.mat')['sadnessVec']
anger = loadmat('W_anger.mat')['angerVec']

data = np.hstack((fear, happy, neutral, sadness, anger))
y = np.array([[i] * 50 for i in range(5)]).flatten()
yy = to_categorical(y)
network = LVQnet(140, 5, [50] * 5)
network.fit(data.T, yy, lr=0.05, iterations=100000)
yp = network.predict_mat(data.T)
yp = np.argmax(yp, axis=1)

confusion_matrix_info(y, yp)
