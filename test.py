import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 生成数据
x = np.arange(5)
y = np.arange(5)
z = np.random.randint(10, size=(5, 5))

# 设置柱子的位置和大小
dx = dy = 0.8
dz = z

# 绘制3D柱状图
for i in range(5):
    for j in range(5):
        ax.bar3d(x[i], y[j], 0, dx, dy, dz[i, j])

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('How2matplotlib.com 3D Bar Plot')

plt.show()