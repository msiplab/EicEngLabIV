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
        expctd_Xobs = None

        # 実現値
        actual_nbfs = phi.nbfs
        actual_width_factor = phi.width_factor
        actual_Xobs = phi.Xobs


        # 評価
        self.assertEqual(actual_nbfs,expctd_nbfs)
        self.assertEqual(actual_width_factor,expctd_width_factor)
        self.assertEqual(actual_Xobs,expctd_Xobs)

    def test_construction_Xobs1d(self):

        # 設定
        M = 10
        nSamples = 100
        rng = np.random.RandomState(1)
        Xobs = rng.rand(nSamples)

        # ターゲットのインスタンス化
        phi = GaussianFeaturesWithKmeans(nbfs=M,Xobs=Xobs)

        # 期待値
        expctd_nbfs = M
        expctd_width_factor = 1.0
        expctd_Xobs = Xobs
    
        # 実現値
        actual_nbfs = phi.nbfs
        actual_width_factor = phi.width_factor
        actual_Xobs = phi.Xobs

        # 評価
        self.assertEqual(actual_nbfs,expctd_nbfs)
        self.assertEqual(actual_width_factor,expctd_width_factor)
        assert_array_equal(actual_Xobs,expctd_Xobs)




        
        

if __name__ == '__main__':
    unittest.main()