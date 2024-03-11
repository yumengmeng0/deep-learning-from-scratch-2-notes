import sys
from dataset import spiral
import matplotlib.pyplot as plt

sys.path.append('..')

x, t = spiral.load_data()
print('x', x.shape)
print('t', t.shape)

# 绘制数据点
N = 100
CLS_NUM = 3
markers = ['o', 'x', '^']
for i in range(CLS_NUM):
    plt.scatter(x[i * N:(i + 1) * N, 0], x[i * N:(i + 1) * N, 1], s=40, marker=markers[i])
plt.show()
