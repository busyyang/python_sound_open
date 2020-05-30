import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

N = 32
nn = [i for i in range(N)]
plt.subplot(3, 1, 1)
plt.stem(np.ones(N))
plt.title('(a)矩形窗')

w = 0.54 - 0.46 * np.cos(np.multiply(nn, 2 * np.pi) / (N - 1))
plt.subplot(3, 1, 2)
plt.stem(w)
plt.title('(b)汉明窗')

w = 0.5 * (1 - np.cos(np.multiply(nn, 2 * np.pi) / (N - 1)))
plt.subplot(3, 1, 3)
plt.stem(w)
plt.title('(c)海宁窗')
# plt.show()
plt.savefig('images/window.png')
plt.close()
