import matplotlib.pyplot as plt
import json
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

def draw(title, data):
    x = []
    y = []
    for m in range(10):
        for mv in range(10):
            x.append(m / 10)
            y.append(mv / 10)


    # 创建图形对象

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 设置 X 轴和 Y 轴的刻度
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))


    # 设置柱子的位置和大小
    dx = dy = 0.03
    dz = data
    z_max = np.max(data)
    z_min = np.min(data)
    ax.set_zlim(z_min, z_max)

    # 创建 colormap 和 Normalize 对象
    norm = Normalize(vmin=z_min, vmax=z_max)
    cmap = cm.viridis  # 可以选择其他 colormap，如 'plasma', 'inferno', 'jet' 等

    # 绘制3D柱状图
    for i in range(10):
        for j in range(10):
            ax.bar3d(x[i*10+j], y[i*10+j], z_min, dx, dy, dz[i*10+j]-z_min)

    ax.set_xlabel('Pixel Mask Ratio')
    ax.set_ylabel('Mask View Ratio')
    ax.set_zlabel('ACC %')
    ax.set_title(title)

    plt.show()



def draw_mask_view(title, data):
    view_data = [[0 for i in range(10)]for j in range(10)]
    for m in range(10):
        for mv in range(10):
            view_data[m][mv]=data[m*10+mv]


    x = [i/10 for i in range(10)]
    # 创建图形对象
    for i in [5]:
        plt.plot(x, view_data[i], label=f'masked_ratio:{i/10}')



    plt.show()




    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlabel('masked_ratio')
    # ax.set_ylabel('mask_view_ratio')
    # ax.set_zlabel('ACC')
    # # 设置X和Y轴的范围为 range(10)/10
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    # # 设置 X 轴和 Y 轴的刻度
    # ax.set_xticks(np.arange(0, 1.1, 0.1))
    # ax.set_yticks(np.arange(0, 1.1, 0.1))
    #
    # z_min = 0.5
    # z_max = 1
    # dz = [item-z_min for item in data]
    # ax.bar3d(x, y, z, dx, dy, dz)
    # ax.set_zlim(z_min, z_max)
    # ax.set_title(title, fontsize=10, loc='left', pad=10)
    #
    # # 调整子图之间的间距
    # plt.subplots_adjust(wspace=0.3, hspace=0.3)
    #
    # # 显示图形
    # plt.show()


def main():
    with open('./grid_eval_coil.json', 'r') as f:
        data = json.load(f)

    key_list = ['consist-cls-acc(50% modal missing)']
    # draw(key_list[0], data[key_list[0]])
    draw_mask_view(key_list[0], data[key_list[0]])


if __name__=='__main__':
    main()