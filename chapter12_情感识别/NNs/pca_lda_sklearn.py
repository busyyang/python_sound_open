import numpy as np
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import matplotlib.pyplot as plt
from sklearn.decomposition import pca
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier


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


if __name__ == '__main__':
    # 载入数据
    fear = loadmat('A_Fear.mat')['fearVec']
    happy = loadmat('F_Happiness.mat')['hapVec']
    neutral = loadmat('N_neutral.mat')['neutralVec']
    sadness = loadmat('T_sadness.mat')['sadnessVec']
    anger = loadmat('W_anger.mat')['angerVec']
    data = np.hstack((fear, happy, neutral, sadness, anger)).T
    y = np.array([[i] * 50 for i in range(5)]).flatten()
    # PCA降维
    p = pca.PCA(10)
    data_re = p.fit(data.T) # fit后返回的是一个结构体，降维后的数据在components_里面
    # 分类器分类
    clf = svm.SVC()
    clf.fit(data_re.components_.T[::2], y[::2])
    yp = clf.predict(data_re.components_.T[::3])
    confusion_matrix_info(y[::3], yp, title='PCA')
    print(classification_report(y[::3], yp, target_names=['fear', 'happy', 'neutr', 'sad', 'anger']))
    print('-' * 100)
    # LDA降维与分类
    lda = LinearDiscriminantAnalysis()
    lda.fit(data[::2], y[::2])
    yp = lda.predict(data[::3])
    confusion_matrix_info(y[::3], yp, title='LDA')
    print(classification_report(y[::3], yp, target_names=['fear', 'happy', 'neutr', 'sad', 'anger']))
