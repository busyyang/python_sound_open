# -*- coding: utf-8 -*-
"""
PNN code from https://github.com/shiluqiang/PNN_python
Modified by Jie Y. 2020/3/24
"""
import numpy as np
import copy
from scipy.io import loadmat
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


def Normalization(data):
    '''样本数据归一化
    input:data(mat):样本特征矩阵
    output:Nor_feature(mat):归一化的样本特征矩阵
    '''
    m, n = np.shape(data)
    Nor_feature = copy.deepcopy(data)
    sample_sum = np.sqrt(np.sum(np.square(data), axis=1))
    for i in range(n):
        Nor_feature[:, i] = Nor_feature[:, i] / sample_sum

    return Nor_feature


def distance(X, Y):
    return np.sum(np.square(X - Y))


def distance_mat(Nor_trainX, Nor_testX):
    '''计算待测试样本与所有训练样本的欧式距离
    input:Nor_trainX(mat):归一化的训练样本
          Nor_testX(mat):归一化的测试样本
    output:Euclidean_D(mat):测试样本与训练样本的距离矩阵
    '''
    m, n = np.shape(Nor_trainX)
    p = np.shape(Nor_testX)[0]
    Euclidean_D = np.mat(np.zeros((p, m)))
    for i in range(p):
        for j in range(m):
            Euclidean_D[i, j] = distance(Nor_testX[i, :], Nor_trainX[j, :])
    return Euclidean_D


def Gauss(Euclidean_D, sigma):
    '''测试样本与训练样本的距离矩阵对应的Gauss矩阵
    input:Euclidean_D(mat):测试样本与训练样本的距离矩阵
          sigma(float):Gauss函数的标准差
    output:Gauss(mat):Gauss矩阵
    '''
    return np.exp(-Euclidean_D / (2 * (sigma ** 2)))


def Prob_mat(Gauss_mat, labelX):
    '''测试样本属于各类的概率和矩阵
    input:Gauss_mat(mat):Gauss矩阵
          labelX(list):训练样本的标签矩阵
    output:Prob_mat(mat):测试样本属于各类的概率矩阵
           label_class(list):类别种类列表
    '''
    ## 找出所有的标签类别
    label_class = list(set(labelX))

    n_class = len(label_class)
    ## 求概率和矩阵
    p, m = np.shape(Gauss_mat)
    Prob = np.mat(np.zeros((p, n_class)))
    for i in range(p):
        for j in range(m):
            for s in range(n_class):
                if labelX[j] == label_class[s]:
                    Prob[i, s] += Gauss_mat[i, j]
                    break
    Prob_mat = copy.deepcopy(Prob)
    Prob_mat = Prob_mat / np.sum(Prob, axis=1)
    return Prob_mat, label_class


def calss_results(Prob, label_class):
    '''分类结果
    input:Prob(mat):测试样本属于各类的概率矩阵
          label_class(list):类别种类列表
    output:results(list):测试样本分类结果
    '''
    arg_prob = np.argmax(Prob, axis=1)  ##类别指针
    results = []
    for i in range(len(arg_prob)):
        results.append(label_class[arg_prob[i, 0]])
    return results


if __name__ == '__main__':
    fear = loadmat('A_Fear.mat')['fearVec']
    happy = loadmat('F_Happiness.mat')['hapVec']
    neutral = loadmat('N_neutral.mat')['neutralVec']
    sadness = loadmat('T_sadness.mat')['sadnessVec']
    anger = loadmat('W_anger.mat')['angerVec']

    data = np.hstack((fear, happy, neutral, sadness, anger)).T
    y = np.array([[i] * 50 for i in range(5)]).flatten()
    # 2、样本数据归一化
    Nor_trainX = Normalization(data)
    Nor_testX = Normalization(data[::3, :])
    # 3、计算Gauss矩阵
    Euclidean_D = distance_mat(Nor_trainX, Nor_testX)
    Gauss_mat = Gauss(Euclidean_D, 0.1)
    Prob, label_class = Prob_mat(Gauss_mat, y)
    # 4、求测试样本的分类
    predict_results = calss_results(Prob, label_class)
    confusion_matrix_info(y[::3], predict_results)
