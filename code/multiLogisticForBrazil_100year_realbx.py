# 巴西实际数据演示

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
r = 0.46
Fm = 2894.166667
Fo = 1621.481481  # 整个模型的起点值（x=0的时候）
vLambda = 0.1639
To = 30
Tm = 30
# vAlpha = 0.75
# cyclePeriod = 15  # 轮伐间隔时间
cycleTime = 70  # 轮伐次数
# sumTime = To + cyclePeriod * cycleTime

valueArray = [g, r, Fm, Fo, vLambda, To, Tm]


def logiMaxFunc(t):
    upperLine = g * Fm * Fo * np.exp(r * vLambda * t)
    downerLine = g * Fm + Fo * (np.exp(r * vLambda * t) - 1)
    Fmi = (upperLine / downerLine)

    return Fmi


def logiMaxCurveFunc():
    # sumTime = To + cyclePeriod[] * cycleTime
    t = np.linspace(0, To, To * 100)
    upperLine = g * Fm * Fo * np.exp(r * vLambda * t)
    downerLine = g * Fm + Fo * (np.exp(r * vLambda * t) - 1)
    FmiCur = (upperLine / downerLine)
    return FmiCur


def logiFunc(initTime, lastTime, initConstant, initValue, Fmi=1):
    lastTime = int(lastTime)
    # print(lastTime)
    t = np.linspace(0, lastTime, lastTime * 100)
    if Fmi == 1:
        Fmi = logiMaxFunc(initTime + lastTime) - initConstant

    if initValue < initConstant:
        return initValue * np.ones(len(t))
    initValue = initValue - initConstant
    # print(Fmi)
    upperLine = g * Fmi * initValue * np.exp(r * vLambda * (t))
    downerLine = g * Fmi + initValue * (np.exp(r * vLambda * (t)) - 1)
    Fi = (upperLine / downerLine) + initConstant


    return Fi


def multiLogiFunc(vAlpha, cyclePeriod):
    Fi0 = logiMaxCurveFunc()

    # for i in range(cycleTime-1):
    #     print(i)
    #     print((To + (i-2) * cyclePeriod) * 100 - 1)
    #     aFi = (logiFunc(To + (i - 1) * cyclePeriod, cyclePeriod, 10, multi_Fi[int((To + (i - 2) * cyclePeriod) * 100 - 1)] * vAlpha))
    #     print(aFi)
    initValue = 1.558e+04 + 3.2045e+05
    initConstant = 260000

    Fi1 = logiFunc(To, cyclePeriod, initConstant, Fi0[To * 100 - 1] * vAlpha)
    Fi2 = logiFunc(To + cyclePeriod, cyclePeriod, initConstant, Fi1[cyclePeriod * 100 - 1] * vAlpha)
    Fi3 = logiFunc(To + 2 * cyclePeriod, cyclePeriod, initConstant, Fi2[cyclePeriod * 100 - 1] * vAlpha)
    Fi4 = logiFunc(To + 3 * cyclePeriod, cyclePeriod, initConstant, Fi3[cyclePeriod * 100 - 1] * vAlpha)
    Fi5 = logiFunc(To + 4 * cyclePeriod, cyclePeriod, initConstant, Fi4[cyclePeriod * 100 - 1] * vAlpha)
    Fi6 = logiFunc(To + 5 * cyclePeriod, cyclePeriod, initConstant, Fi5[cyclePeriod * 100 - 1] * vAlpha)
    Fi7 = logiFunc(To + 6 * cyclePeriod, cyclePeriod, initConstant, Fi6[cyclePeriod * 100 - 1] * vAlpha)
    Fi8 = logiFunc(To + 7 * cyclePeriod, cyclePeriod, initConstant, Fi7[cyclePeriod * 100 - 1] * vAlpha)
    Fi9 = logiFunc(To + 8 * cyclePeriod, cyclePeriod, initConstant, Fi8[cyclePeriod * 100 - 1] * vAlpha)
    Fi10 = logiFunc(To + 9 * cyclePeriod, cyclePeriod, initConstant, Fi9[cyclePeriod * 100 - 1] * vAlpha)
    Fi11 = logiFunc(To + 10 * cyclePeriod, cyclePeriod, initConstant, Fi10[cyclePeriod * 100 - 1] * vAlpha)
    Fi12 = logiFunc(To + 11 * cyclePeriod, cyclePeriod, initConstant, Fi11[cyclePeriod * 100 - 1] * vAlpha)
    Fi13 = logiFunc(To + 12 * cyclePeriod, cyclePeriod, initConstant, Fi12[cyclePeriod * 100 - 1] * vAlpha)
    Fi14 = logiFunc(To + 13 * cyclePeriod, cyclePeriod, initConstant, Fi13[cyclePeriod * 100 - 1] * vAlpha)
    Fi15 = logiFunc(To + 14 * cyclePeriod, cyclePeriod, initConstant, Fi14[cyclePeriod * 100 - 1] * vAlpha)
    Fi16 = logiFunc(To + 15 * cyclePeriod, cyclePeriod, initConstant, Fi15[cyclePeriod * 100 - 1] * vAlpha)
    Fi17 = logiFunc(To + 16 * cyclePeriod, cyclePeriod, initConstant, Fi16[cyclePeriod * 100 - 1] * vAlpha)
    Fi18 = logiFunc(To + 17 * cyclePeriod, cyclePeriod, initConstant, Fi17[cyclePeriod * 100 - 1] * vAlpha)
    Fi19 = logiFunc(To + 18 * cyclePeriod, cyclePeriod, initConstant, Fi18[cyclePeriod * 100 - 1] * vAlpha)
    Fi20 = logiFunc(To + 19 * cyclePeriod, cyclePeriod, initConstant, Fi19[cyclePeriod * 100 - 1] * vAlpha)
    Fi21 = logiFunc(To + 20 * cyclePeriod, cyclePeriod, initConstant, Fi20[cyclePeriod * 100 - 1] * vAlpha)
    Fi22 = logiFunc(To + 21 * cyclePeriod, cyclePeriod, initConstant, Fi21[cyclePeriod * 100 - 1] * vAlpha)
    Fi23 = logiFunc(To + 22 * cyclePeriod, cyclePeriod, initConstant, Fi22[cyclePeriod * 100 - 1] * vAlpha)
    Fi24 = logiFunc(To + 23 * cyclePeriod, cyclePeriod, initConstant, Fi23[cyclePeriod * 100 - 1] * vAlpha)
    Fi25 = logiFunc(To + 24 * cyclePeriod, cyclePeriod, initConstant, Fi24[cyclePeriod * 100 - 1] * vAlpha)
    Fi26 = logiFunc(To + 25 * cyclePeriod, cyclePeriod, initConstant, Fi25[cyclePeriod * 100 - 1] * vAlpha)
    Fi27 = logiFunc(To + 26 * cyclePeriod, cyclePeriod, initConstant, Fi26[cyclePeriod * 100 - 1] * vAlpha)
    Fi28 = logiFunc(To + 27 * cyclePeriod, cyclePeriod, initConstant, Fi27[cyclePeriod * 100 - 1] * vAlpha)
    Fi29 = logiFunc(To + 28 * cyclePeriod, cyclePeriod, initConstant, Fi28[cyclePeriod * 100 - 1] * vAlpha)
    Fi30 = logiFunc(To + 29 * cyclePeriod, cyclePeriod, initConstant, Fi29[cyclePeriod * 100 - 1] * vAlpha)
    Fi31 = logiFunc(To + 30 * cyclePeriod, cyclePeriod, initConstant, Fi30[cyclePeriod * 100 - 1] * vAlpha)
    Fi32 = logiFunc(To + 31 * cyclePeriod, cyclePeriod, initConstant, Fi31[cyclePeriod * 100 - 1] * vAlpha)
    Fi33 = logiFunc(To + 32 * cyclePeriod, cyclePeriod, initConstant, Fi32[cyclePeriod * 100 - 1] * vAlpha)
    Fi34 = logiFunc(To + 33 * cyclePeriod, cyclePeriod, initConstant, Fi33[cyclePeriod * 100 - 1] * vAlpha)
    Fi35 = logiFunc(To + 34 * cyclePeriod, cyclePeriod, initConstant, Fi34[cyclePeriod * 100 - 1] * vAlpha)
    Fi36 = logiFunc(To + 35 * cyclePeriod, cyclePeriod, initConstant, Fi35[cyclePeriod * 100 - 1] * vAlpha)
    Fi37 = logiFunc(To + 36 * cyclePeriod, cyclePeriod, initConstant, Fi36[cyclePeriod * 100 - 1] * vAlpha)
    Fi38 = logiFunc(To + 37 * cyclePeriod, cyclePeriod, initConstant, Fi37[cyclePeriod * 100 - 1] * vAlpha)
    Fi39 = logiFunc(To + 38 * cyclePeriod, cyclePeriod, initConstant, Fi38[cyclePeriod * 100 - 1] * vAlpha)
    Fi40 = logiFunc(To + 39 * cyclePeriod, cyclePeriod, initConstant, Fi39[cyclePeriod * 100 - 1] * vAlpha)
    Fi41 = logiFunc(To + 40 * cyclePeriod, cyclePeriod, initConstant, Fi40[cyclePeriod * 100 - 1] * vAlpha)
    Fi42 = logiFunc(To + 41 * cyclePeriod, cyclePeriod, initConstant, Fi41[cyclePeriod * 100 - 1] * vAlpha)
    Fi43 = logiFunc(To + 42 * cyclePeriod, cyclePeriod, initConstant, Fi42[cyclePeriod * 100 - 1] * vAlpha)
    Fi44 = logiFunc(To + 43 * cyclePeriod, cyclePeriod, initConstant, Fi43[cyclePeriod * 100 - 1] * vAlpha)
    Fi45 = logiFunc(To + 44 * cyclePeriod, cyclePeriod, initConstant, Fi44[cyclePeriod * 100 - 1] * vAlpha)
    Fi46 = logiFunc(To + 45 * cyclePeriod, cyclePeriod, initConstant, Fi45[cyclePeriod * 100 - 1] * vAlpha)
    Fi47 = logiFunc(To + 46 * cyclePeriod, cyclePeriod, initConstant, Fi46[cyclePeriod * 100 - 1] * vAlpha)
    Fi48 = logiFunc(To + 47 * cyclePeriod, cyclePeriod, initConstant, Fi47[cyclePeriod * 100 - 1] * vAlpha)
    Fi49 = logiFunc(To + 48 * cyclePeriod, cyclePeriod, initConstant, Fi48[cyclePeriod * 100 - 1] * vAlpha)
    Fi50 = logiFunc(To + 49 * cyclePeriod, cyclePeriod, initConstant, Fi49[cyclePeriod * 100 - 1] * vAlpha)
    Fi51 = logiFunc(To + 50 * cyclePeriod, cyclePeriod, initConstant, Fi50[cyclePeriod * 100 - 1] * vAlpha)
    Fi52 = logiFunc(To + 51 * cyclePeriod, cyclePeriod, initConstant, Fi51[cyclePeriod * 100 - 1] * vAlpha)
    Fi53 = logiFunc(To + 52 * cyclePeriod, cyclePeriod, initConstant, Fi52[cyclePeriod * 100 - 1] * vAlpha)
    Fi54 = logiFunc(To + 53 * cyclePeriod, cyclePeriod, initConstant, Fi53[cyclePeriod * 100 - 1] * vAlpha)
    Fi55 = logiFunc(To + 54 * cyclePeriod, cyclePeriod, initConstant, Fi54[cyclePeriod * 100 - 1] * vAlpha)
    Fi56 = logiFunc(To + 55 * cyclePeriod, cyclePeriod, initConstant, Fi55[cyclePeriod * 100 - 1] * vAlpha)
    Fi57 = logiFunc(To + 56 * cyclePeriod, cyclePeriod, initConstant, Fi56[cyclePeriod * 100 - 1] * vAlpha)
    Fi58 = logiFunc(To + 57 * cyclePeriod, cyclePeriod, initConstant, Fi57[cyclePeriod * 100 - 1] * vAlpha)
    Fi59 = logiFunc(To + 58 * cyclePeriod, cyclePeriod, initConstant, Fi58[cyclePeriod * 100 - 1] * vAlpha)
    Fi60 = logiFunc(To + 59 * cyclePeriod, cyclePeriod, initConstant, Fi59[cyclePeriod * 100 - 1] * vAlpha)
    Fi61 = logiFunc(To + 60 * cyclePeriod, cyclePeriod, initConstant, Fi60[cyclePeriod * 100 - 1] * vAlpha)
    Fi62 = logiFunc(To + 61 * cyclePeriod, cyclePeriod, initConstant, Fi61[cyclePeriod * 100 - 1] * vAlpha)
    Fi63 = logiFunc(To + 62 * cyclePeriod, cyclePeriod, initConstant, Fi62[cyclePeriod * 100 - 1] * vAlpha)
    Fi64 = logiFunc(To + 63 * cyclePeriod, cyclePeriod, initConstant, Fi63[cyclePeriod * 100 - 1] * vAlpha)
    Fi65 = logiFunc(To + 64 * cyclePeriod, cyclePeriod, initConstant, Fi64[cyclePeriod * 100 - 1] * vAlpha)
    Fi66 = logiFunc(To + 65 * cyclePeriod, cyclePeriod, initConstant, Fi65[cyclePeriod * 100 - 1] * vAlpha)
    Fi67 = logiFunc(To + 66 * cyclePeriod, cyclePeriod, initConstant, Fi66[cyclePeriod * 100 - 1] * vAlpha)
    Fi68 = logiFunc(To + 67 * cyclePeriod, cyclePeriod, initConstant, Fi67[cyclePeriod * 100 - 1] * vAlpha)
    Fi69 = logiFunc(To + 68 * cyclePeriod, cyclePeriod, initConstant, Fi68[cyclePeriod * 100 - 1] * vAlpha)
    Fi70 = logiFunc(To + 69 * cyclePeriod, cyclePeriod, initConstant, Fi69[cyclePeriod * 100 - 1] * vAlpha)
    # Fi1 = logiFunc(To, cyclePeriod, initValue, initConstant, 90000)
    # Fi2 = logiFunc(To + cyclePeriod, cyclePeriod, initValue, Fi1[cyclePeriod * 100 - 1] * vAlpha)
    # Fi3 = logiFunc(To + 2 * cyclePeriod, cyclePeriod, initValue, Fi2[cyclePeriod * 100 - 1] * vAlpha)
    # Fi4 = logiFunc(To + 3 * cyclePeriod, cyclePeriod, initValue, Fi3[cyclePeriod * 100 - 1] * vAlpha)
    # Fi5 = logiFunc(To + 4 * cyclePeriod, cyclePeriod, initValue, Fi4[cyclePeriod * 100 - 1] * vAlpha)
    # Fi6 = logiFunc(To + 5 * cyclePeriod, cyclePeriod, initValue, Fi5[cyclePeriod * 100 - 1] * vAlpha)
    # Fi7 = logiFunc(To + 6 * cyclePeriod, cyclePeriod, initValue, Fi6[cyclePeriod * 100 - 1] * vAlpha)
    # Fi8 = logiFunc(To + 7 * cyclePeriod, cyclePeriod, initValue, Fi7[cyclePeriod * 100 - 1] * vAlpha)
    # Fi9 = logiFunc(To + 8 * cyclePeriod, cyclePeriod, initValue, Fi8[cyclePeriod * 100 - 1] * vAlpha)
    # Fi10 = logiFunc(To + 9 * cyclePeriod, cyclePeriod, initValue, Fi9[cyclePeriod * 100 - 1] * vAlpha)
    # Fi11 = logiFunc(To + 10 * cyclePeriod, cyclePeriod, initValue, Fi10[cyclePeriod * 100 - 1] * vAlpha)
    # Fi12 = logiFunc(To + 11 * cyclePeriod, cyclePeriod, initValue, Fi11[cyclePeriod * 100 - 1] * vAlpha)
    # Fi13 = logiFunc(To + 12 * cyclePeriod, cyclePeriod, initValue, Fi12[cyclePeriod * 100 - 1] * vAlpha)
    # Fi14 = logiFunc(To + 13 * cyclePeriod, cyclePeriod, initValue, Fi13[cyclePeriod * 100 - 1] * vAlpha)
    # Fi15 = logiFunc(To + 14 * cyclePeriod, cyclePeriod, initValue, Fi14[cyclePeriod * 100 - 1] * vAlpha)
    # Fi16 = logiFunc(To + 15 * cyclePeriod, cyclePeriod, initValue, Fi15[cyclePeriod * 100 - 1] * vAlpha)
    # Fi17 = logiFunc(To + 16 * cyclePeriod, cyclePeriod, initValue, Fi16[cyclePeriod * 100 - 1] * vAlpha)
    # Fi1 = logiFunc(To, cyclePeriod, 10, Fi0[To * 100 - 1] * vAlpha)
    # Fi2 = logiFunc(To + cyclePeriod, cyclePeriod, 10, Fi1[cyclePeriod * 100 - 1] * vAlpha)
    # Fi3 = logiFunc(To + 2 * cyclePeriod, cyclePeriod, 10, Fi2[cyclePeriod * 100 - 1] * vAlpha)
    # Fi4 = logiFunc(To + 3 * cyclePeriod, cyclePeriod, 10, Fi3[cyclePeriod * 100 - 1] * vAlpha)
    # Fi5 = logiFunc(To + 4 * cyclePeriod, cyclePeriod, 10, Fi4[cyclePeriod * 100 - 1] * vAlpha)
    # Fi6 = logiFunc(To + 5 * cyclePeriod, cyclePeriod, 10, Fi5[cyclePeriod * 100 - 1] * vAlpha)
    # Fi7 = logiFunc(To + 6 * cyclePeriod, cyclePeriod, 10, Fi6[cyclePeriod * 100 - 1] * vAlpha)
    # Fi8 = logiFunc(To + 7 * cyclePeriod, cyclePeriod, 10, Fi7[cyclePeriod * 100 - 1] * vAlpha)
    # Fi9 = logiFunc(To + 8 * cyclePeriod, cyclePeriod, 10, Fi8[cyclePeriod * 100 - 1] * vAlpha)
    # Fi10 = logiFunc(To + 9 * cyclePeriod, cyclePeriod, 10, Fi9[cyclePeriod * 100 - 1] * vAlpha)
    #
    # print(np.shape(Fi0))
    # print(np.shape(Fi1))
    # print(np.shape(Fi2))
    # print(np.shape(Fi3))
    # print(np.shape(Fi4))
    # print(np.shape(Fi5))
    # print(np.shape(Fi6))
    multi_Fi = np.hstack((Fi0, Fi1, Fi2, Fi3, Fi4, Fi5, Fi6, Fi7, Fi8, Fi9, Fi10, Fi11, Fi12, Fi13, Fi14, Fi15, Fi16,
                          Fi17, Fi18, Fi19, Fi20, Fi21, Fi22, Fi23, Fi24, Fi25, Fi26, Fi27, Fi28, Fi29, Fi30, Fi31,
                          Fi32, Fi33, Fi34, Fi35, Fi36, Fi3, Fi38, Fi39, Fi40, Fi41, Fi42, Fi43, Fi44, Fi45, Fi46,
                          Fi47, Fi48, Fi49, Fi50, Fi51, Fi52, Fi53, Fi54, Fi55, Fi56, Fi57, Fi58, Fi59, Fi60, Fi61,
                          Fi62, Fi63, Fi64, Fi65, Fi66, Fi67, Fi68, Fi69, Fi70))
    # print(np.shape(multi_Fi))
    return multi_Fi


# 画图
def ploter(columeNum, rowNum, vAlpha, cyclePeriod, ax, count):
    F = multiLogiFunc(vAlpha, cyclePeriod)
    sumTime = To + cyclePeriod * cycleTime
    t = np.linspace(0, sumTime, sumTime * 100)

    Fwood = WoodFunc(t, cyclePeriod, F)

    F_ADDwood = F + Fwood
    # F_ADDwood = Fwood
    FIntegral = []
    for i in range(sumTime*100):
        FIntegral.append((scipy.integrate.trapz(F_ADDwood[:i+1], t[:i+1]) / i * 100))
    # print(np.shape(FIntegral))


    # plt.title('Curves of fungal length over time, taking into account the effect of environmental moisture')

    # plt.xlabel('t / years')
    # plt.ylabel('v / m3')


    # print(np.shape(t))
    # print(np.shape(F))
    t = t[:10000]
    F = F[:10000]
    Fwood = Fwood[:10000]
    F_ADDwood = F_ADDwood[:10000]
    FIntegral = FIntegral[:10000]
    plt.plot(t, F, label="Cycle")
    plt.plot(t, Fwood, label="Wood")
    plt.plot(t, F_ADDwood, label="Cycle+Wood")
    plt.plot(t, logiMaxFunc(t), label="not cut")
    # plt.figure()
    plt.plot(t, FIntegral, label="TotalCS")
    # plt.plot(t, F)
    # plt.plot(t, logiMaxFunc(t))
    plt.xlim(0, 100)
    plt.ylim(0, 600000)
    # plt.plot(time, num[:, 1], label="Specie2")
    # plt.legend()
    # for ci in range(columeNum):
    #     if count != rowNum * columeNum - ci:
    #         ax.set_xticks([])
    # for ri in range(rowNum):
    #     if count != rowNum * ri:
    #         ax.set_yticks([])

    #
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    # ax.spines['bottom'].set_color('none')
    # ax.spines['left'].set_color('none')
    return F_ADDwood[9999]

def WoodFunc(t, cyclePeriod, F):
    # t = np.linspace(0, sumTime, sumTime * 100)
    # To 轮伐开始的年数
    # cycleTime 轮伐次数
    sumTime = To + cyclePeriod * cycleTime
    beginCorrupt = 20  # 开始腐烂
    endCorrupt = 31  # 结束腐烂
    WoodFuncValues = np.zeros(len(t))
    WoodRecValues = np.zeros(len(t))  # 方形
    WoodTriValues = np.zeros(len(t))  # 三角形

    for i in range(cycleTime):
        # print((To + i * cyclePeriod) * 100)
        l = 0.7 * (F[(To + i*cyclePeriod) * 100-1] - F[(To + i*cyclePeriod) * 100])
        # print(l)

        a = np.zeros((To + cyclePeriod * i) * 100)
        b = l * np.ones(beginCorrupt * 100)
        if (cyclePeriod * cycleTime - cyclePeriod * i - beginCorrupt) < 0:
            RecPlus = np.hstack((a, b))
            RecPlus = RecPlus[:(sumTime * 100)]
        else:
            c = np.zeros((cyclePeriod * cycleTime - cyclePeriod * i - beginCorrupt) * 100)
            RecPlus = np.hstack((a, b, c))
        WoodRecValues = WoodRecValues + RecPlus

        a = np.zeros((To + cyclePeriod * i + beginCorrupt) * 100)
        b = np.ones((endCorrupt - beginCorrupt) * 100)
        for j in range(len(b)):
            b[j] = b[0] - j * b[0] / len(b)
        b = b * l
        if (cyclePeriod * cycleTime - cyclePeriod * i - endCorrupt) <0:
            TriPlus = np.hstack((a, b))
            TriPlus = TriPlus[:(sumTime * 100)]
        else:
            c = np.zeros((cyclePeriod * cycleTime - cyclePeriod * i - endCorrupt) * 100)
            TriPlus = np.hstack((a, b, c))
        WoodTriValues = WoodTriValues + TriPlus
    WoodFuncValues = WoodRecValues + WoodTriValues

    return WoodFuncValues



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
    vAlpha = [0.8, 0.9]
    cyclePeriod = [15, 25]

    count = 0

    AGB_100 = np.zeros((len(vAlpha), len(cyclePeriod)))
    for i in range(len(vAlpha)):
        for j in range(len(cyclePeriod)):
            sns.set_style("whitegrid")
            ax = plt.subplot(len(vAlpha), len(cyclePeriod), 1 + count)
            count = count + 1
            print(count)
            AGB_100[i][j] = ploter(len(vAlpha), len(cyclePeriod), vAlpha[i], cyclePeriod[j], ax, count)
            # curveFs = np.vstack((curveFs, curveF))
    # print(1)
    # print(1)
    # print(1)
    # print(1)
    # print(1)
    # print(AGB_100)

    # sumTime = []
    # integrals = []  # 用于存储积分
    # for j in range(len(cyclePeriod)):
    #     # print(np.shape(sumTime))
    #     sumTime.append(To + cyclePeriod[j] * cycleTime)
    #     # print(sumTime[j])
    #     integrals = []  # 用于存储积分
    #     t = np.linspace(0, sumTime[j], sumTime[j] * 100)
    #     print(np.shape(sumTime))
    #     print(np.shape(t))
    #     for integralsNum in range(sumTime[j]):
    #         integrals.append(scipy.integrate.trapz(sumTime, t))
    #     plt.plot(t, integrals)



    plt.show()
    # integrals = []  # 用于存储积分
    #
    # for i in range(len(sumTime)):  # 计算梯形的面积，由于是累加，所以是切片"i+1"
    #     integrals.append(scipy.integrate.trapz(num[:i + 1, 0], time[:i + 1]))
    #
    # sns.set_style("whitegrid")
    # plt.plot(np.linspace(0, sumTime, sumTime * 100), integrals, label="Fungi B")
    # plt.xlabel('t / days')
    # plt.ylabel('r')
    # plt.xlim(0)
    # plt.ylim(0, 70000)
    # plt.legend()
    # ax = plt.gca()  # 获取当前图像的坐标轴信息
    # ax.yaxis.get_major_formatter().set_powerlimits((0, 1))  # 将坐标轴的base number设置为一位。
    # plt.show()
