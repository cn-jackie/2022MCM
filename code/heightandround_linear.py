import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import pandas as pd
import seaborn as sns


def f_1(x, A, B):
    return A * x + B


if __name__ == '__main__':
    # file_data = pd.read_csv('heightandround.csv')
    file_data = pd.read_csv('DBHandHeight.csv')
    sns.set_style("white")

    plt.figure()
    # 拟合点
    x0 = file_data["DBH"]  # cm
    y0 = file_data["Height"]  # m

    # 绘制散点
    plt.scatter(x0[:], y0[:], 3)

    # 直线拟合与绘制
    A1, B1 = optimize.curve_fit(f_1, x0, y0)[0]
    x1 = np.arange(0, 30, 0.01)  # 30和75要对应x0的两个端点，0.01为步长
    y1 = A1 * x1 + B1
    plt.plot(x1, y1)
    print(A1)
    print(B1)
    # 0.45464550259998915
    # 1.9496592267295678
    # sns.jointplot(data=file_data, kind='reg')
    plt.show()
