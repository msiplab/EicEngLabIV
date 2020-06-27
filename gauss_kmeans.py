"""GaussianFeaturesWithKmenasモジュール

        *
        *

    Todo:

        *
        *

Copyright (c) 2020, Shogo MURAMATSU, All rights reserved.
"""
import numpy as np
import pandas as pd 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class GaussianFeaturesWithKmeans(BaseEstimator, TransformerMixin):
    """K-means利用した2段階推定法によるガウス基底関数

        scikit-learn LinearRegression推定器に渡すためのガウス基底関数による
        特徴量への変換クラス    

    Attributes:
        centers_ (属性の型): 属性の説明
        widths_ (:obj:`属性の型`): 属性の説明.

    References:

        * 小西貞則「多変量解析入門－線形から非線形へ－」岩波書店
        * Jake VanderPlas（菊池彰訳）「Pythonデータサイエンスハンドブック」オライリージャパン

    """

    def __init__(self, nbfs, width_factor=1.0, prekmeans=True):
        """GaussianFeaturesWithKMeansのコンストラクタ

        Args:
            nbfs (int): 基底数
            width_factor (float): 基底関数の広がり係数
            prekmeans (bool): K-平均法により前処理

        Returns: 
           GaussianFeaturesWithKMeans: ガウス基底関数オブジェクト

        Raises:
            例外の名前: 例外の説明 (例 : 引数が指定されていない場合に発生 )

        Examples:

            関数の使い方について記載

            >>> 
               

        Note:

            (c) Copyright, Shogo MURAMATSU, All rights reserved
        """
        self.nbfs = nbfs
        self.width_factor = width_factor
        self.prekmeans = prekmeans

    @staticmethod
    def _gauss_basis(x, y, width, axis=None):
        arg = (x - y) / (width + 1e-300)
        return np.exp(-0.5 * np.sum(arg ** 2, axis))

    def fit(self, X, y=None):
        ndims = X.shape[1]
        if self.prekmeans:
            scaler = StandardScaler()         
            kmeans = KMeans(n_clusters=self.nbfs,random_state=0)
            X = scaler.fit_transform(X)
            kmeans.fit(X)

            labels = kmeans.predict(X).reshape(-1,1)
            clusters = pd.DataFrame(np.concatenate((labels,X),axis=1)).groupby([0])
            if ndims == 1:
                self.centers_ = scaler.inverse_transform(kmeans.cluster_centers_).reshape(-1,)
                self.widths_ = (self.width_factor * scaler.scale_ * clusters.std(ddof=0)).to_numpy().reshape(-1,)
            else:
                self.centers_ = scaler.inverse_transform(kmeans.cluster_centers_).reshape(-1,ndims,1).transpose(2,1,0) 
                self.widths_ = (self.width_factor * scaler.scale_ * clusters.std(ddof=0)).to_numpy().reshape(-1,ndims,1).transpose(2,1,0)
        else:
            if ndims == 1:
                self.centers_ = np.linspace(X.min(), X.max(), self.nbfs) 
                self.widths_ = self.width_factor * (self.centers_[1] - self.centers_[0]) * np.ones(self.nbfs)
            else:
                raise Exception('prekmeansを True に設定してください．重回帰はK平均法による事前推定のみ対応しています．')        
        
        return self

    def transform(self, X):
        return self._gauss_basis(X[:,:, np.newaxis], self.centers_,
                                self.widths_, axis=1)

if __name__ == '__main__':
    import matplotlib.pyplot as plt 
    import seaborn as sns; sns.set()
    rng = np.random.RandomState()

    '''
    # 単回帰
    M = 3
    width_factor = 1.0
    nSamples = 100
    x = 10 * rng.rand(nSamples)
    y = np.sin(x) + 0.1 * rng.randn(nSamples)

    phi1 = GaussianFeaturesWithKmeans(M,width_factor=width_factor,prekmeans=True)
    gauss_model1 = make_pipeline(phi1,LinearRegression())
    gauss_model1.fit(x.reshape(-1,1), y)

    fig1, ax1 = plt.subplots()
    nPoints = 1000
    xfit = np.linspace(0, 10, nPoints)
    yfit = gauss_model1.predict(xfit.reshape(-1,1))
    ax1.scatter(x,y)
    ax1.plot(xfit,yfit,color='red')
    ax1.set_xlim(0,10)
    plt.show()
    '''

    # 重回帰
    M = 8
    width_factor = 1.0
    nSamples = 1000
    rng = np.random.RandomState(1)    
    x1 = 0.1 * rng.rand(nSamples)
    x2 = 10 * rng.rand(nSamples)
    X = np.concatenate((x1.reshape(-1,1),x2.reshape(-1,1)),axis=1)
    y = np.sin(100*x1) + np.cos(x2) + 0.01 * rng.randn(nSamples)
    phi2 = GaussianFeaturesWithKmeans(M,width_factor=width_factor,prekmeans=True) 
    gauss_model2 = make_pipeline(phi2,LinearRegression())
    gauss_model2.fit(X,y)

    nPoints = 100
    xfit1,xfit2 = np.meshgrid(np.linspace(0, 0.1, nPoints),
                                np.linspace(0, 10, nPoints))
    Xfit  = np.concatenate([xfit1.reshape(-1,1),xfit2.reshape(-1,1)],axis=1)
    yfit  = gauss_model2.predict(Xfit).reshape(xfit1.shape)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(x1,x2,y)
    ax.plot_wireframe(xfit1,xfit2,yfit, color = 'red')
    ax.set_xlim(0,0.1)
    ax.set_ylim(0,10)
    plt.show()

    
