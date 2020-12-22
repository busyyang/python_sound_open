"""
LDA算法将数据投影到新的轴上去
来源：https://blog.csdn.net/z962013489/article/details/79871789
和：  https://blog.csdn.net/z962013489/article/details/79918758
"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


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


class LDA:
    def __init__(self, num):
        self.num = num

    def fit(self, x, y):
        classes = list(set(y))
        mus = np.zeros((len(classes), x.shape[1]))
        Sw = np.zeros((x.shape[1], x.shape[1]))  # 计算类内散度矩阵
        Sb = np.zeros((x.shape[1], x.shape[1]))  # 计算类间散度矩阵
        i = 0
        x_classed = {}
        for n in classes:
            data = x[np.where(y == n)[0]]  # 取出当前分类的所有数据
            x_classed[i] = data
            mus[i] = np.mean(data, axis=0)  # 当前分类的均值
            data -= mus[i]  # 去中心化数据
            Sw += np.matmul(data.T, data)  # 计算类内散度矩阵
            i += 1
        # 计算类间距离
        for i in range(len(classes)):
            dd = mus[i] - np.mean(x, axis=0)
            Sb += len(x_classed[i]) * np.matmul(dd.reshape(x.shape[1], -1), dd.reshape(-1, x.shape[1]))
        ## 或者这样计算Sb
        # St = np.matmul((x - np.mean(x, axis=0)).T, x - np.mean(x, axis=0))
        # Sb = St - Sw
        eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))  # 计算Sw-1*Sb的特征值和特征矩阵
        sorted_indices = np.argsort(eig_vals)[::-1]  # 特征值排序并倒序过来，最大值在前
        topk_eig_vecs = eig_vecs[:, sorted_indices[:self.num]]  # 提取前k个特征向量
        self.topk_eig_vecs = topk_eig_vecs
        return topk_eig_vecs


class PCA:
    def __init__(self, num):
        self.num = num

    def fit(self, x):
        """
        得到特征值最大的前K个投影方向的投影，并将投影方向保存到self.topk_eig_vecs中
        :param x: 目标数据(n_samples,n_features)
        :return:
        """
        # 去中心化
        x -= np.mean(x, axis=0)
        # 求协方差矩阵
        covMat = np.cov(x, rowvar=0)
        # 特征分解得到特征向量与特征值
        eigVals, eigVects = np.linalg.eig(np.mat(covMat))  # 求特征值和特征向量,特征向量是按列放的
        # 取前K个特征值最大的向量
        eigValIndice = np.argsort(eigVals)[::-1]  # 对特征值从大到小排序
        topk_eig_vecs = eigVects[:, eigValIndice[:self.num]]  # 提取前k个特征向量
        self.topk_eig_vecs = topk_eig_vecs
        return np.dot(x, topk_eig_vecs)

    def pca_new(self, x):
        """
        将新数据进行投影
        :param x:目标数据(n_samples,n_features)
        :return:
        """
        return np.dot(x, self.topk_eig_vecs)


if '__main__' == __name__:
    # 载入数据
    fear = loadmat('A_Fear.mat')['fearVec']
    happy = loadmat('F_Happiness.mat')['hapVec']
    neutral = loadmat('N_neutral.mat')['neutralVec']
    sadness = loadmat('T_sadness.mat')['sadnessVec']
    anger = loadmat('W_anger.mat')['angerVec']
    x = np.hstack((fear, happy, neutral, sadness, anger)).T
    y = np.array([[i] * 50 for i in range(5)]).flatten()
    ## 使用自定义的LDA降维
    lda = LDA(num=10)
    W = lda.fit(x, y)
    X_new = np.dot(x, W)
    plt.subplot(121)
    plt.scatter(X_new[:, 0], X_new[:, 1], marker='o', c=y)
    plt.title('myLDA')
    ## 使用sklearn LDA降维
    lda2 = LinearDiscriminantAnalysis(n_components=10)
    lda2.fit(x, y)
    X_new_2 = lda2.transform(x)
    plt.subplot(122)
    plt.scatter(X_new_2[:, 0], X_new_2[:, 1], marker='o', c=y)
    plt.title('sklearn-LDA')

    # 降维后分类看看分类器分类
    clf = svm.SVC()
    clf.fit(X_new[::2], y[::2])
    yp = clf.predict(X_new[::3])
    confusion_matrix_info(y[::3], yp, title='MyLDA')

    pca = PCA(num=10)
    xx = pca.fit(x)
    # 这是新的数据，与x是同分布的，计算投影后的特征空间
    x_new = pca.pca_new(x[::3])
    clf = svm.SVC()
    clf.fit(xx, y)
    ypp = clf.predict(x_new)
    confusion_matrix_info(y[::3], ypp, title='myPCA')
