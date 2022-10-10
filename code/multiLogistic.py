# 种群竞争模型

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint  # 用以求常微分
import scipy
from scipy.integrate import simps
import seaborn as sns
from sympy import *

plt.rcParams['figure.dpi'] = 100  # 绘图的dpi
plt.rcParams['axes.unicode_minus'] = False  # 正常显示正负号

# 两个物种的参数
g = 1
r = 1
Fm = 150000
Fo = 1  # 整个模型的起点值（x=0的时候）
vLambda = 1
To = 15
Tm = 30
cyclePeriod = 15  # 轮伐间隔时间
cycleTime = 10  # 轮伐次数
sumTime = To + cyclePeriod * cycleTime
vAlpha = 0.7
valueArray = [g, r, Fm, Fo, vLambda, To, Tm]


def logiMaxFunc(t):
    upperLine = g * Fm * Fo * np.exp(r * vLambda * t)
    downerLine = g * Fm + Fo * (np.exp(r * vLambda * t) - 1)
    Fmi = (upperLine / downerLine) + Fo

    return Fmi

def logiMaxCurveFunc():
    t = np.linspace(0, To, To * 100)
    upperLine = g * Fm * Fo * np.exp(r * vLambda * t)
    downerLine = g * Fm + Fo * (np.exp(r * vLambda * t) - 1)
    FmiCur = (upperLine / downerLine) + Fo
    return FmiCur


def logiFunc(initTime, lastTime, initValue, initConstant):
    t = np.linspace(0, lastTime, lastTime * 100)
    Fmi = logiMaxFunc(initTime + lastTime) - initConstant
    print(Fmi)
    upperLine = g * Fmi * initValue * np.exp(r * vLambda * t)
    downerLine = g * Fmi + initValue * (np.exp(r * vLambda * t) - 1)
    Fi = (upperLine / downerLine) + initConstant

    return Fi


def multiLogiFunc():
    Fi0 = logiMaxCurveFunc()

    # for i in range(cycleTime-1):
    #     print(i)
    #     print((To + (i-2) * cyclePeriod) * 100 - 1)
    #     aFi = (logiFunc(To + (i - 1) * cyclePeriod, cyclePeriod, 10, multi_Fi[int((To + (i - 2) * cyclePeriod) * 100 - 1)] * vAlpha))
    #     print(aFi)

    Fi1 = logiFunc(To, cyclePeriod, 10, Fi0[To * 100 - 1] * vAlpha)
    Fi2 = logiFunc(To + cyclePeriod, cyclePeriod, 10, Fi1[cyclePeriod * 100 - 1] * vAlpha)
    Fi3 = logiFunc(To + 2 * cyclePeriod, cyclePeriod, 10, Fi2[cyclePeriod * 100 - 1] * vAlpha)
    Fi4 = logiFunc(To + 3 * cyclePeriod, cyclePeriod, 10, Fi3[cyclePeriod * 100 - 1] * vAlpha)
    Fi5 = logiFunc(To + 4 * cyclePeriod, cyclePeriod, 10, Fi4[cyclePeriod * 100 - 1] * vAlpha)
    Fi6 = logiFunc(To + 5 * cyclePeriod, cyclePeriod, 10, Fi5[cyclePeriod * 100 - 1] * vAlpha)
    Fi7 = logiFunc(To + 6 * cyclePeriod, cyclePeriod, 10, Fi6[cyclePeriod * 100 - 1] * vAlpha)
    Fi8 = logiFunc(To + 7 * cyclePeriod, cyclePeriod, 10, Fi7[cyclePeriod * 100 - 1] * vAlpha)
    Fi9 = logiFunc(To + 8 * cyclePeriod, cyclePeriod, 10, Fi8[cyclePeriod * 100 - 1] * vAlpha)
    Fi10 = logiFunc(To + 9 * cyclePeriod, cyclePeriod, 10, Fi9[cyclePeriod * 100 - 1] * vAlpha)
    print(np.shape(Fi0))
    print(np.shape(Fi1))
    print(np.shape(Fi2))
    print(np.shape(Fi3))
    print(np.shape(Fi4))
    print(np.shape(Fi5))
    print(np.shape(Fi6))
    multi_Fi = np.hstack((Fi0, Fi1, Fi2, Fi3, Fi4, Fi5, Fi6, Fi7, Fi8, Fi9, Fi10))
    print(np.shape(multi_Fi))
    return multi_Fi


# 画图
def ploter():
    F = multiLogiFunc()

    sns.set_style("whitegrid")
    # plt.title('Curves of fungal length over time, taking into account the effect of environmental moisture')

    plt.xlabel('t / years')
    plt.ylabel('v / m3')
    t = np.linspace(0, sumTime, sumTime * 100)
    print(np.shape(t))
    print(np.shape(F))
    plt.plot(t, F, label="Cycle")
    plt.plot(t, logiMaxFunc(t), label="not cut")
    plt.xlim(0)
    plt.ylim(0)
    # plt.plot(time, num[:, 1], label="Specie2")
    plt.legend()
    plt.show()


# 运行
if __name__ == '__main__':
    # time1 = np.linspace(0, 75, 1000)  # 时间为200个单位，均分为1000份
    # time2 = np.linspace(75, 150, 1000)
    # time3 = np.linspace(150, 225, 1000)
    # init1 = np.array([type_x1[0], type_x2[0]])
    # num1 = odeint(propagate, y0=init1, t=time1, args=(type_x1, type_x2))  # num是两条线的纵坐标值
    # init2 = num1[999]
    # num2 = odeint(propagate1, y0=init2, t=time2, args=(type_x1, type_x2))  # num是两条线的纵坐标值
    # init3 = num2[999]
    # num3 = odeint(propagate2, y0=init3, t=time3, args=(type_x1, type_x2))

    # num = np.vstack((num1, num2, num3))
    # time = (*time1, *time2, *time3)
    ploter()

    # integrals = []  # 用于存储积分
    #
    # for i in range(len(num[:, 0])):  # 计算梯形的面积，由于是累加，所以是切片"i+1"
    #     integrals.append(scipy.integrate.trapz(num[:i + 1, 0], time[:i + 1]))
    #
    # sns.set_style("whitegrid")
    # plt.plot(time, integrals, label="Fungi B")
    # plt.xlabel('t / days')
    # plt.ylabel('r')
    # plt.xlim(0)
    # plt.ylim(0, 70000)
    # plt.legend()
    # ax = plt.gca()  # 获取当前图像的坐标轴信息
    # ax.yaxis.get_major_formatter().set_powerlimits((0, 1))  # 将坐标轴的base number设置为一位。
    # plt.show()
