#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 03/03/18

@author: XXX
"""

import unittest
import numpy as np

from RecSysFramework.DataManager.Reader import Movielens20MReader

from RecSysFramework.DataManager.DatasetPostprocessing import KCore
from RecSysFramework.DataManager.DatasetPostprocessing import ImplicitURM
from RecSysFramework.DataManager.DatasetPostprocessing import UserSample


class DataPostprocessingTestCase(unittest.TestCase):

    def setUp(self):
        self.dataset = Movielens20MReader().load_data()


    def matrix_contained(self, new_URM, old_URM):
        nnz_row, nnz_col = new_URM.nonzero()
        for t in range(len(nnz_row)):
            self.assertNotEqual(old_URM[nnz_row[t], nnz_col[t]], 0, "Value {} in new URM not found in old URM".format(t))



class DataPostprocessingImplicitURMTest(DataPostprocessingTestCase):

    def test_full(self):
        postprocessing = ImplicitURM()
        dataset = postprocessing.apply(self.dataset)

        old_URM = self.dataset.get_URM()
        new_URM = dataset.get_URM()

        old_URM.sort_indices()
        new_URM.sort_indices()

        self.assertTrue(np.all(old_URM.indptr == new_URM.indptr) and np.all(old_URM.indices == new_URM.indices),
                        "Old and new URM have different shapes after implicitization")

        self.assertTrue(np.all(new_URM.data == 1), "New URM is not implicit")


    def test_thresholded(self):
        threshold = 3
        postprocessing = ImplicitURM(min_rating_threshold=threshold)
        dataset = postprocessing.apply(self.dataset)

        old_URM = self.dataset.get_URM()
        new_URM = dataset.get_URM()

        old_URM.data[old_URM.data <= threshold] = 0.0
        old_URM.eliminate_zeros()

        old_URM.sort_indices()
        new_URM.sort_indices()

        self.assertTrue(np.all(old_URM.indptr == new_URM.indptr) and np.all(old_URM.indices == new_URM.indices),
                        "Old and new URM have different non-zero indices after implicitization")

        self.assertTrue(np.all(new_URM.data == 1), "New URM is not implicit")


    def test_save_data(self):
        postprocessing = ImplicitURM(min_rating_threshold=3)
        dataset = postprocessing.apply(self.dataset)
        dataset.save_data()



class DataPostprocessingKCoreTest(DataPostprocessingTestCase):

    def test_asymmetric_reshaped(self):
        user_k_core = 3
        item_k_core = 5
        postprocessing = KCore(user_k_core=user_k_core, item_k_core=item_k_core, reshape=True)
        dataset = postprocessing.apply(self.dataset)

        old_URM = self.dataset.get_URM()
        new_URM = dataset.get_URM()

        self.assertTrue(np.all(np.ediff1d(new_URM.tocsc().indptr) >= item_k_core), "K core not respected over items")
        self.assertTrue(np.all(np.ediff1d(new_URM.tocsr().indptr) >= user_k_core), "K core not respected over users")


    def test_symmetric_not_reshaped(self):
        user_k_core = item_k_core = 5
        postprocessing = KCore(user_k_core=user_k_core, item_k_core=item_k_core, reshape=False)
        dataset = postprocessing.apply(self.dataset)

        old_URM = self.dataset.get_URM()
        new_URM = dataset.get_URM()

        self.matrix_contained(new_URM, old_URM)


    def test_save_data(self):
        user_k_core = item_k_core = 5
        postprocessing = KCore(user_k_core=user_k_core, item_k_core=item_k_core, reshape=True)
        dataset = postprocessing.apply(self.dataset)
        dataset.save_data()



class DataPostprocessingUserSample(DataPostprocessingTestCase):

    def test_half(self):
        postprocessing = UserSample(user_quota=0.5)
        dataset = postprocessing.apply(self.dataset)


    def test_no_changes(self):
        postprocessing = UserSample(user_quota=1.0)
        dataset = postprocessing.apply(self.dataset)

        old_URM = self.dataset.get_URM()
        new_URM = dataset.get_URM()

        old_URM.sort_indices()
        new_URM.sort_indices()

        self.assertTrue(np.all(old_URM.indptr == new_URM.indptr) and np.all(old_URM.indices == new_URM.indices),
                        "Old and new URM have different non-zero indices after 1.0 quota user sample")


    def test_save_data(self):
        postprocessing = UserSample(user_quota=0.5)
        dataset = postprocessing.apply(self.dataset)
        dataset.save_data()



if __name__ == '__main__':

    unittest.main()
