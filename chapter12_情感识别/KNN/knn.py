import numpy as np
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


def get_most_label(result):
    rst = {}
    for r in result:
        if r not in rst.keys():
            rst[r] = 1
        else:
            rst[r] += 1
    m = sorted(rst.items(), key=lambda x: x[1], reverse=True)
    return m[0][0]


K = 9

fear = loadmat('A_Fear.mat')['fearVec']
happy = loadmat('F_Happiness.mat')['hapVec']
neutral = loadmat('N_neutral.mat')['neutralVec']
sadness = loadmat('T_sadness.mat')['sadnessVec']
anger = loadmat('W_anger.mat')['angerVec']

data = np.hstack((fear, happy, neutral, sadness, anger))
y = np.array([[i] * 50 for i in range(5)]).flatten()
per = np.random.permutation(250)
data_train = data[:, per[:180]]
label_train = y[per[:180]]
data_test = data[:, per[180:]]
label_test = y[per[180:]]
label_pred = np.zeros(250 - 180)
j = 0
for test in data_test.T:
    scores = np.zeros(len(data_train.T))
    for i in range(len(data_train.T)):
        scores[i] = np.sum(np.power(test - data_train[:, i], 2))
    pos = np.argsort(scores)[:K]
    result = label_train[pos]
    label = get_most_label(result)
    label_pred[j] = label
    j += 1

confusion_matrix_info(label_test, label_pred)
