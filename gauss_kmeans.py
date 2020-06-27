"""GaussianFeaturesWithKmenasモジュール

        *
        *

    Todo:

        *
        *

Copyright (c) 2020, Shogo MURAMATSU, All rights reserved.
"""
import sys
import numpy as np
import pandas as pd 
from sklearn.cluster import KMeans
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
        if any(width==0):
            raise ValueError('基底関数が多すぎるようです')
        else:
            arg = (x - y) / width
        return np.exp(-0.5 * np.sum(arg ** 2, axis))

    def fit(self, X, y=None):
        # print('fit')        
        if self.prekmeans:
            ndims = X.shape[1]
            kmeans = KMeans(n_clusters=self.nbfs,random_state=0)
            kmeans.fit(X)
            labels = kmeans.predict(X).reshape(-1,1)
            clusters = pd.DataFrame(np.concatenate((labels,X),axis=1)).groupby([0])
            if ndims == 1:
                self.centers_ = kmeans.cluster_centers_.reshape(-1,)
            else:
                self.centers_ = kmeans.cluster_centers_.reshape(-1,ndims)                
            self.widths_ = self.width_factor * np.sqrt(clusters.var(ddof=0).sum(axis=1)).to_numpy()
        else:
            self.centers_ = np.linspace(X.min(), X.max(), self.nbfs) 
            self.widths_ = self.width_factor * (self.centers_[1] - self.centers_[0]) * np.ones(self.nbfs)
            #print(self.centers_.shape)
            #print(self.widths_.shape)       
        return self

    def transform(self, X):
        # print('transform')
        return self._gauss_basis(X[:,:, np.newaxis], self.centers_,
                                self.widths_, axis=1)

if __name__ == '__main__':
    import matplotlib.pyplot as plt 
    import seaborn as sns; sns.set()
    
    # 単回帰
    M = 10
    nSamples = 50
    rng = np.random.RandomState(1)
    x = 10 * rng.rand(nSamples)
    y = np.sin(x) + 0.1 * rng.randn(nSamples)

    phi = GaussianFeaturesWithKmeans(M,width_factor=1.0,prekmeans=True)
    gauss_model = make_pipeline(phi,LinearRegression())
    gauss_model.fit(x[:, np.newaxis], y)
    xfit = np.linspace(0, 10, 1000)
    yfit = gauss_model.predict(xfit[:,np.newaxis])
    
    fig, ax = plt.subplots()
    ax.scatter(x,y)
    ax.plot(xfit,yfit)
    ax.set_xlim(0,10)
    plt.show()
    
    # 重回帰
            #
        #from sklearn.linear_model import LinearRegression 
        #from sklearn.pipeline import make_pipeline
        #mymodel = make_pipeline(phi,LinearRegression())
        #mymodel.fit(X[:,:,np.newaxis],y)
        #xfit1,xfit2 = np.meshgrid(np.linspace(minx1,maxx1,nPoints),
        #                         np.linspace(minx2,maxx2,nPoints))
        #Xfit  = np.concatenate([xfit1.reshape(-1,1),xfit2.reshape(-1,1)],axis=1)
        #yfit  = mymodel.predict(Xfit).reshape(xfit1.shape)
        #ax.scatter(x1,x2,y)
        #ax.plot_wireframe(xfit1,xfit2,yfit, color = 'red')
        #yfit = mymodel.predict(xfit[:,np.newaxis])
