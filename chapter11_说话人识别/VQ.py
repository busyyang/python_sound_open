import numpy as np


def dis(u, xi):
    """
    计算欧式距离
    :param u:
    :param xi:
    :return:
    """
    k = u.shape[1]
    xi = xi.squeeze()
    dis = np.zeros(k)
    for i in range(k):
        ui = u[:, i]
        dis[i] = np.sum(np.power(xi - ui, 2))
    return dis


def lbg(x, k):
    """
    完成lbg均值聚类算法
    :param x:为row*col矩阵，每一列为一个样本，每个样本有row个元素
    :param k:返回k个分类
    :return:
    """
    row, col = x.shape
    epislon = 0.03
    delta = 0.01
    u = np.mean(x, axis=1).reshape(x.shape[0], -1)  # 第一个聚类中心，总体均值
    v = {}
    for i3 in range(int(np.log2(k))):
        u = np.hstack((u * (1 - epislon), u * (1 + epislon)))
        D = 0
        DD = 1
        while abs((D - DD) / (DD + epislon)) > delta:
            DD = D
            for i in range(pow(2, i3 + 1)):  # 初始化
                vv = {}
                vv['num'] = 0
                vv['ele'] = np.zeros(row)
                v[i] = vv
            for i in range(col):  # 计算第i个样本
                distance = dis(u, x[:, i])
                pos = np.argmin(distance)
                v[pos]['num'] += 1
                if v[pos]['num'] == 1:
                    v[pos]['ele'] = x[:, i]
                else:
                    v[pos]['ele'] = np.vstack((v[pos]['ele'], x[:, i]))
            for i in range(pow(2, i3 + 1)):
                u[:, i] = np.mean(v[i]['ele'], axis=0)
                for m in range(v[i]['ele'].shape[0]):
                    D += np.sum(np.power(v[i]['ele'][m] - u[:, i], 2))
    for i in range(k):
        v[i]['mean'] = u[:, i]
    return v
