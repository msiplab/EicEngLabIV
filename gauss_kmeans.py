"""GaussianFeaturesWithKmenasモジュール

        *
        *

    Todo:

        *
        *

Copyright (c) 2020, Shogo MURAMATSU, All rights reserved.
"""
import numpy as np
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

    def __init__(self, nbfs, width_factor=1.0, Xobs=None):
        """GaussianFeaturesWithKMeansのコンストラクタ

        Args:
            nbfs (int): 基底数
            width_factor (float): 基底関数の広がり係数
            Xobs (np.ndarray): 引数の説明

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

        self.Xobs = Xobs
    
    @staticmethod
    def _gauss_basis(x, y, width, axis=None):
        arg = (x - y) / width
        return np.exp(-0.5 * np.sum(arg ** 2, axis))

    def fit(self, X, y=None):
        # create N centers spread along the data range
        self.centers_ = np.linspace(X.min(), X.max(), self.nbfs)
        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
        return self

    def transform(self, X):
        return self._gauss_basis(X[:,:, np.newaxis], self.centers_,
                                self.width_, axis=1)
    