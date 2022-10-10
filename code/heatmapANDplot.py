import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    file_data = pd.read_csv('DBHandHeight.csv')

    df = file_data.head(40)

    x0 = file_data["DBH"]  # cm
    y0 = file_data["Height"]  # m
    sns.jointplot(x0, y0)
    plt.show()
    plt.figure()
    sns.set_style("white")

    sns.heatmap(df.corr())
    sns.pairplot(df)
    plt.show()

