import numpy as np


def k_means(centres, data, kiter):
    dim, data_sz = data.T.shape
    ncentres = centres.shape[0]
    per = np.random.permutation(data_sz)
    perm = per[:ncentres]
    centres = data[perm, :]
    id = np.eye(ncentres)
    for n in range(kiter):
        old_centres = centres
        d2 = np.multiply(np.ones((ncentres, 1)), np.sum(np.power(data, 2), axis=1, keepdims=True).T).T + \
             np.multiply(np.ones((data_sz, 1)), np.sum(np.power(centres, 2), axis=1, keepdims=True).T) - \
             2 * np.matmul(data, centres.T)
        pos = np.argmin(d2, axis=1)
        post = id[pos, :]
        num_points = np.sum(post, axis=0)
        for j in range(ncentres):
            if num_points[j] > 0:
                p = np.where(post[:, j] > 0.5)[0]
                centres[j, :] = np.sum(data[p, :], axis=0) / num_points[j]
        e = np.sum(d2[pos, :])
        if n > 0:
            if np.max(np.abs(centres - old_centres)) < 0.0001 and abs(old_e - e) < 0.0001:
                return centres, post
        old_e = e
    return centres, post


def gmm_init(ncentres, data, kiter, covar_type):
    """
    GMM模型初始化
    :param ncentres:混合模型数目
    :param data:训练数据
    :param kiter:kmeans的迭代次数
    :param covar_type:
    :return:
    """
    mix = {}
    dim, data_sz = data.T.shape
    mix['priors'] = [1 / ncentres for _ in range(ncentres)]
    centres = np.random.rand(ncentres, dim)
    mix['covars'] = np.tile(np.eye(dim), (ncentres, 1, 1))
    GMM_WIDTH = 1
    centres, post = k_means(centres, data, kiter)
    cluster_sizes = np.max(np.sum(post, axis=1), axis=0)
    priors = cluster_sizes / np.sum(cluster_sizes)
    for j in range(ncentres):
        p = np.where(post[:, j] > 0.5)[0]
        c = data[p, :]
        diffs = c - np.multiply(np.ones((c.shape[0], 1)), centres[j, :])
        mix['covars'][j, ...] = np.matmul(diffs.T, diffs) / (c.shape[1] + 1e-7)
        if np.linalg.matrix_rank(mix['covars'][j, ...]) < dim:
            mix['covars'][j, ...] += GMM_WIDTH * np.eye(dim)
    mix['ncentres'] = ncentres
    mix['covar_type'] = covar_type
    mix['centres'] = centres
    return mix


def calcpost(mix, x):
    pass


def gmm_em(mix, x, emiter):
    """

    :param mix:
    :param x:
    :param emiter:
    :return:
    """
    dim, data_sz = x.T.shape
    init_covars = mix['covars']
    MIN_COVAR = 0.001
    for cnt in range(emiter):
        # --- E step: 计算充分统计量 ---
        post, act = calcpost(mix, x)
