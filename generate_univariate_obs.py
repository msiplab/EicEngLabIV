import numpy as np
from numpy.random import *
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import japanize_matplotlib 
sns.set(font="IPAexGothic")

def generate_univariate_obs(filename):
    nSamples = 200
    sigmaw = 0.05
    a = 20.0
    b = 0.5
    
    x = rand(nSamples)
    f = lambda x : 1./(1.+np.exp(-a*(x-b)))
    w = sigmaw*randn(nSamples)
    y = f(x) + w

    df = pd.DataFrame({ 'XDATA':x, 'YDATA':y })
    df.to_csv(filename,index=None)

if __name__ == '__main__':
    filename = './data/sample01_01.csv'
    generate_univariate_obs(filename)
    dataset = pd.read_csv(filename)

    # 説明変数の設定
    x = dataset.iloc[:,0].values.reshape(-1,1) 

    # 目的変数の設定
    y = dataset.iloc[:,1].values.reshape(-1,1)

    # 散布図の範囲情報の抽出
    minx = np.min(x)
    maxx = np.max(x)
    rx = maxx-minx
    cx = 0.5*rx
    miny = np.min(y)
    maxy = np.max(y)
    ry = maxy-miny
    cy = 0.5*ry

    # 散布図の表示
    fig, ax = plt.subplots()
    ax.scatter(x,y)
    ax.set_title("サンプル１")
    ax.set_xlabel(dataset.columns[0])
    ax.set_ylabel(dataset.columns[1])
    ax.set_xlim([minx-0.1*rx,maxx+0.1*rx])
    ax.set_ylim([miny-0.1*ry,maxy+0.1*ry])
    ax.grid(True)
    plt.show()
