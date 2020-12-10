#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 02/08/18

@author: XXX
"""


import unittest

from RecSysFramework.DataManager.Reader import Movielens1MReader

from RecSysFramework.DataManager.Splitter import Holdout, ColdItemsHoldout
from RecSysFramework.DataManager.Splitter import WarmItemsKFold, ColdItemsKFold
from RecSysFramework.DataManager.Splitter import LeaveKOut

from RecSysFramework.Utils import invert_dictionary


class SplitterTestCase(unittest.TestCase):

    def setUp(self):
        self.reader = Movielens1MReader()
        self.dataset = self.reader.load_data()


    def matrix_contained(self, new_URM, old_URM, new_mapper, old_mapper):

        new_mapper = invert_dictionary(new_mapper[0]), invert_dictionary(new_mapper[1])

        old_URM.eliminate_zeros()
        nnz_row, nnz_col = new_URM.nonzero()

        for t in range(len(nnz_row)):
            row, col = old_mapper[0][new_mapper[0][nnz_row[t]]], old_mapper[1][new_mapper[1][nnz_col[t]]]
            self.assertEqual(old_URM[row, col], new_URM[nnz_row[t], nnz_col[t]],
                             "Value {} in ({}, {}) of new URM is zero in old URM"
                             .format(new_URM[nnz_row[t], nnz_col[t]], nnz_row[t], nnz_col[t]))


    def split_contained(self, URM_train, URM_test, URM_valid, new_mapper, sum_equals_original=False):

        URM_all = self.dataset.get_URM()
        old_mapper = self.dataset.get_URM_mapper()

        URM_sum = URM_train + URM_test
        if URM_valid is not None:
            URM_sum += URM_valid

        if sum_equals_original:
            self.assertEqual(len(URM_sum.data), len(URM_all.data),
                             "Original URM and sum of train, test and validation have different number of entries")

        self.matrix_contained(URM_train, URM_all, new_mapper, old_mapper)
        self.matrix_contained(URM_test, URM_all, new_mapper, old_mapper)
        if URM_valid is not None:
            self.matrix_contained(URM_valid, URM_all, new_mapper, old_mapper)
        self.matrix_contained(URM_sum, URM_all, new_mapper, old_mapper)



class SplitterHoldoutTest(SplitterTestCase):

    splits = [(0.6, 0.2, 0.2), (0.8, 0.2, 0.0)]

    def test_holdout(self):

        print("Holdout Test")

        for trio in self.splits:

            train_perc, test_perc, validation_perc = trio
            splitter = Holdout(train_perc=train_perc, test_perc=test_perc, validation_perc=validation_perc)

            if validation_perc > 0.0:
                train, test, validation = splitter.split(self.dataset)
                URM_valid = validation.get_URM()
            else:
                train, test = splitter.split(self.dataset)
                URM_valid = None

            URM_train = train.get_URM()
            URM_test = test.get_URM()

            self.split_contained(URM_train, URM_test, URM_valid, train.get_URM_mapper())


    def test_cold_holdout(self):

        print("Cold Items Holdout Test")

        for trio in self.splits:

            train_perc, test_perc, validation_perc = trio
            splitter = ColdItemsHoldout(train_perc=train_perc, test_perc=test_perc, validation_perc=validation_perc)

            if validation_perc > 0.0:
                train, test, validation = splitter.split(self.dataset)
                URM_valid = validation.get_URM()
            else:
                train, test = splitter.split(self.dataset)
                URM_valid = None

            URM_train = train.get_URM()
            URM_test = test.get_URM()

            self.split_contained(URM_train, URM_test, URM_valid, train.get_URM_mapper())


    def test_load_and_save_data(self):
        print("Holdout load and save Test")
        splitter = Holdout(train_perc=0.6, test_perc=0.2, validation_perc=0.2)
        train, test, validation = splitter.load_split(self.reader)
        splitter.save_split([train, test, validation])



class SplitterKFoldTest(SplitterTestCase):

    def test_kfold(self):

        print("K Fold Test")

        n_folds = 5
        splitter = WarmItemsKFold(n_folds=n_folds)
        counter = 0

        for train, test in splitter.split(self.dataset):

            URM_train = train.get_URM()
            URM_test = test.get_URM()

            self.split_contained(URM_train, URM_test, None, train.get_URM_mapper())
            counter += 1

        self.assertEqual(counter, n_folds, "Number of folds generated not consistent")


    def test_cold_kfold(self):

        print("Cold Items K Fold Test")

        n_folds = 5
        splitter = ColdItemsKFold(n_folds=n_folds)
        counter = 0

        for train, test in splitter.split(self.dataset):
            URM_train = train.get_URM()
            URM_test = test.get_URM()

            self.split_contained(URM_train, URM_test, None, train.get_URM_mapper())
            counter += 1

        self.assertEqual(counter, n_folds, "Number of folds generated not consistent")


    def test_load_and_save_data(self):
        print("K Fold load and save Test")
        splitter = WarmItemsKFold(n_folds=5)
        for train, test in splitter.load_split(self.reader):
            splitter.save_split([train, test])



class SplitterLeaveKOutTest(SplitterTestCase):

    def test_leavekout(self):

        print("Leave K Out Test")

        for k_value, with_validation in [(1, False), (5, True)]:

            splitter = LeaveKOut(k_value=k_value, with_validation=with_validation)

            if with_validation:
                train, test, validation = splitter.split(self.dataset)
                URM_valid = validation.get_URM()
            else:
                train, test = splitter.split(self.dataset)
                URM_valid = None

            URM_train = train.get_URM()
            URM_test = test.get_URM()

            self.split_contained(URM_train, URM_test, URM_valid, train.get_URM_mapper())


    def test_load_and_save_data(self):
        print("Leave K Out load and save Test")
        splitter = LeaveKOut(k_value=1, with_validation=True)
        train, test, validation = splitter.load_split(self.reader)
        splitter.save_split([train, test, validation])



if __name__ == '__main__':
    unittest.main()
