"""GaussianFeaturesWithKmenasテストケース

        *
        *

    Todo:

        *
        *

    Copyright (c) 2020, Shogo MURAMATSU, All rights reserved.
"""
import unittest
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from numpy.testing import assert_array_equal
from gauss_kmeans import GaussianFeaturesWithKmeans

class GaussianFeaturesWithKMeansTestCase(unittest.TestCase):
    """GaussianFeaturesWithKMeansのテストケース

     GaussianFeaturesWithKMeansクラスのユニットテストを定義

    """

    def test_construction(self):
        """コンストラクタのテスト"""

        # 設定
        M = 10

        # ターゲットのインスタンス化
        phi = GaussianFeaturesWithKmeans(nbfs=M)

        # 期待値
        expctd_nbfs = M
        expctd_width_factor = 1.0
        expctd_prekmeans = True
        
        # 実現値
        actual_nbfs = phi.nbfs
        actual_width_factor = phi.width_factor
        actual_prekmeans = phi.prekmeans

        # 評価
        self.assertEqual(actual_nbfs,expctd_nbfs)
        self.assertEqual(actual_width_factor,expctd_width_factor)
        self.assertEqual(actual_prekmeans,expctd_prekmeans)

    def test_x1d_wo_kmeans(self):
        """単変量のテスト"""

        # 設定
        M = 10
        nSamples = 50
        rng = np.random.RandomState(1)
        x = 10 * rng.rand(nSamples)
        X = x.reshape(-1,1)

        # ターゲットのインスタンス化
        phi = GaussianFeaturesWithKmeans(nbfs=M,prekmeans=False)
        phi.fit(X)

        # 期待値
        expctd_nbfs = M
        expctd_width_factor = 1.0
        expctd_prekmeans = False
        expctd_centers = np.linspace(X.min(), X.max(), M)
        expctd_widths = (expctd_centers[1] - expctd_centers[0])*np.ones(M)
 
        # 実現値
        actual_nbfs = phi.nbfs
        actual_width_factor = phi.width_factor
        actual_prekmeans = phi.prekmeans
        actual_centers = phi.centers_
        actual_widths = phi.widths_

        # 評価
        self.assertEqual(actual_nbfs,expctd_nbfs)
        self.assertEqual(actual_width_factor,expctd_width_factor)
        self.assertEqual(actual_prekmeans,expctd_prekmeans)
        assert_array_equal(actual_widths,expctd_widths,'widths')
        assert_array_equal(actual_centers,expctd_centers,'centers')

    def test_x1d_w_kmeans(self):
        """K平均法前処理のテスト"""

        # 設定
        M = 10
        nSamples = 50
        rng = np.random.RandomState(1)
        x = 10 * rng.rand(nSamples)
        y = np.sin(x) + 0.1 * rng.randn(nSamples)
        X = x.reshape(-1,1)
        kmeans = KMeans(n_clusters=M,random_state=0)
        kmeans.fit(X)

        # ターゲットのインスタンス化
        phi = GaussianFeaturesWithKmeans(nbfs=M,prekmeans=True)
        phi.fit(X)

        # 期待値
        expctd_nbfs = M
        expctd_width_factor = 1.0
        expctd_prekmeans = True
        expctd_centers = kmeans.cluster_centers_.reshape(-1,)
        labels = kmeans.predict(X).reshape(-1,1)
        clusters = pd.DataFrame(np.concatenate((labels,X),axis=1)).groupby([0])
        expctd_widths = clusters.std(ddof=0).to_numpy().reshape(-1,)

        # 実現値
        actual_nbfs = phi.nbfs
        actual_width_factor = phi.width_factor
        actual_prekmeans = phi.prekmeans
        actual_centers = phi.centers_
        actual_widths = phi.widths_

        # 評価
        self.assertEqual(actual_nbfs,expctd_nbfs)
        self.assertEqual(actual_width_factor,expctd_width_factor)
        self.assertEqual(actual_prekmeans,expctd_prekmeans)
        assert_array_equal(actual_widths,expctd_widths,'widths')
        assert_array_equal(actual_centers,expctd_centers,'centers')

    def test_x2d_w_kmeans(self):
        """K平均法前処理，二変量のテスト"""

           # 設定
        M = 10
        nSamples = 50
        rng = np.random.RandomState(1)
        x1 = 10 * rng.rand(nSamples)
        x2 = 10 * rng.rand(nSamples)
        X = np.concatenate((x1.reshape(-1,1),x2.reshape(-1,1)),axis=1)
        y = np.sin(x1) + np.cos(x2) + 0.1 * rng.randn(nSamples)
        kmeans = KMeans(n_clusters=M,random_state=0)
        kmeans.fit(X)

        # ターゲットのインスタンス化
        phi = GaussianFeaturesWithKmeans(nbfs=M,prekmeans=True)
        phi.fit(X)

        # 期待値
        expctd_nbfs = M
        expctd_width_factor = 1.0
        expctd_prekmeans = True
        expctd_centers = kmeans.cluster_centers_.reshape(-1,2,1).transpose(2,1,0)
        labels = kmeans.predict(X).reshape(-1,1)
        clusters = pd.DataFrame(np.concatenate((labels,X),axis=1)).groupby([0])
        expctd_widths = np.sqrt(clusters.var(ddof=0).sum(axis=1)).to_numpy()
        
        # 実現値
        actual_nbfs = phi.nbfs
        actual_width_factor = phi.width_factor
        actual_prekmeans = phi.prekmeans
        actual_centers = phi.centers_
        actual_widths = phi.widths_

        # 評価
        self.assertEqual(actual_nbfs,expctd_nbfs)
        self.assertEqual(actual_width_factor,expctd_width_factor)
        self.assertEqual(actual_prekmeans,expctd_prekmeans)
        assert_array_equal(actual_widths,expctd_widths,'widths')
        assert_array_equal(actual_centers,expctd_centers,'centers')

if __name__ == '__main__':
    unittest.main()