#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 03/03/18

@author: XXX
"""

import unittest
import numpy as np

from RecSysFramework.DataManager.Reader import BookCrossingReader
from RecSysFramework.DataManager.Reader import LastFMHetrec2011Reader
from RecSysFramework.DataManager.Reader import Movielens1MReader
from RecSysFramework.DataManager.Reader import Movielens20MReader
from RecSysFramework.DataManager.Reader import YelpReader


class DataReaderTest(unittest.TestCase):

    def setUp(self):
        self.dataReader_list = [
            BookCrossingReader,
            LastFMHetrec2011Reader,
            Movielens1MReader,
            Movielens20MReader,
        ]


    def test_datareader_load_and_save_data(self):
        for dataReader_class in self.dataReader_list:
            dr = dataReader_class(reload_from_original_data=True)
            dataset = dr.load_data()
            dataset.save_data()



class DataReaderUtilsTest(unittest.TestCase):

    def test_reconcile_mapper_with_removed_tokens(self):

        from RecSysFramework.DataManager.Utils import reconcile_mapper_with_removed_tokens

        original_mapper = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}

        reconciled_mapper = reconcile_mapper_with_removed_tokens(original_mapper.copy(), [0])
        self.assertEqual(reconciled_mapper, {"b": 0, "c": 1, "d": 2, "e": 3}, "reconciled_mapper not matching control")

        reconciled_mapper = reconcile_mapper_with_removed_tokens(original_mapper.copy(), [4])
        self.assertEqual(reconciled_mapper, {"a": 0, "b": 1, "c": 2, "d": 3}, "reconciled_mapper not matching control")

        reconciled_mapper = reconcile_mapper_with_removed_tokens(original_mapper.copy(), [0, 2])
        self.assertEqual(reconciled_mapper, {"b": 0, "d": 1, "e": 2}, "reconciled_mapper not matching control")

        reconciled_mapper = reconcile_mapper_with_removed_tokens(original_mapper.copy(), [0, 1, 2, 3, 4])
        self.assertEqual(reconciled_mapper, {}, "reconciled_mapper not matching control")


    def test_split_big_CSR_in_columns(self):

        import scipy.sparse as sps
        from RecSysFramework.DataManager.Utils import split_big_CSR_in_columns

        for num_split in [2, 3, 5, 12]:

            sparse_matrix = sps.random(50, 12, density=0.1, format='csr')

            split_list = split_big_CSR_in_columns(sparse_matrix, num_split=num_split)
            split_rebuilt = sps.hstack(split_list)

            self.assertTrue(np.allclose(sparse_matrix.toarray(), split_rebuilt.toarray()), "split_rebuilt not matching sparse_matrix")



if __name__ == '__main__':

    unittest.main()
