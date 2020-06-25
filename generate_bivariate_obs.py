import numpy as np
from numpy.random import *
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import japanize_matplotlib 
sns.set(font="IPAexGothic")

def generate_bivariate_obs(filename):
    nPoints = [ 10, 20 ]
    a = 500
    b1 = 100
    b2 = 0.01
    x1max = 0.01
    x2max = 800
    sigmaw = 10
    lsx1 = np.linspace(0,x1max,nPoints[0])
    lsx2 = np.linspace(0,x2max,nPoints[1])

    x1,x2 = np.meshgrid(lsx1,lsx2)
    f = lambda x : a * np.exp(-b1*x[:,0]) * (1 - np.exp(-b2*x[:,1]))
    Xdata = np.concatenate([x1.reshape(-1,1), x2.reshape(-1,1)],axis=1)
    w = sigmaw*randn(np.prod(x1.shape))
    y = f(Xdata) + w

    df = pd.DataFrame(y.reshape(nPoints[1],nPoints[0]).T,index=lsx1,columns=lsx2)
    df.to_csv(filename)

if __name__ == '__main__':
    filename = './data/sample01_02.csv'
    generate_bivariate_obs(filename)
    dataset = pd.read_csv(filename)
    
    # 
    lsx1 = dataset.columns[1:].astype(float).values
    lsx2 = dataset.iloc[:,0].astype(float).values
    y = dataset.iloc[0:,1:].astype(float).values

    # 説明変数の設定（メッシュグリッド）
    x1,x2 = np.meshgrid(lsx1,lsx2) 
    print(x1)
    print(x2)
    print(y)

    # 散布図の範囲情報の抽出
    minx1 = np.min(x1)
    maxx1 = np.max(x1)
    rx1 = maxx1-minx1
    cx1 = 0.5*rx1
    minx2 = np.min(x2)
    maxx2 = np.max(x2)
    rx2 = maxx2-minx2
    cx2 = 0.5*rx2

    # 三次元散布図の表示
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(x1,x2,y)
    ax.set_title('サンプル２')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    ax.set_xlim([minx1-0.1*rx1,maxx1+0.1*rx1])
    ax.set_ylim([minx2-0.1*rx2,maxx2+0.1*rx2])
    ax.view_init(elev=30,azim=30) # 見やすいように設定
    ax.grid(True)
    plt.show()
