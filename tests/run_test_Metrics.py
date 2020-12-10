#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 30/09/17

@author: XXX
"""

import numpy as np
import unittest


class MetricsTestCase(unittest.TestCase):

    def test_Gini_Index(self):

        from RecSysFramework.Evaluation.Metrics import Gini_Diversity

        n_items = 1000

        gini_index = Gini_Diversity(n_items, ignore_items=np.array([]))

        gini_index.recommended_counter = np.ones(n_items)
        self.assertTrue(np.isclose(1.0, gini_index.get_metric_value(), atol=1e-2), "Gini_Index metric incorrect")

        gini_index.recommended_counter = np.ones(n_items)*1e-12
        gini_index.recommended_counter[0] = 1.0
        self.assertTrue(np.isclose(0.0, gini_index.get_metric_value(), atol=1e-2), "Gini_Index metric incorrect")


    def test_Shannon_Entropy(self):

        from RecSysFramework.Evaluation.Metrics import Shannon_Entropy

        n_items = 1000

        shannon_entropy = Shannon_Entropy(n_items, ignore_items=np.array([]))

        shannon_entropy.recommended_counter = np.ones(n_items)
        self.assertTrue(np.isclose(9.96, shannon_entropy.get_metric_value(), atol=1e-2), "metric incorrect")

        shannon_entropy.recommended_counter = np.zeros(n_items)
        shannon_entropy.recommended_counter[0] = 1.0
        self.assertTrue(np.isclose(0.0, shannon_entropy.get_metric_value(), atol=1e-3), "metric incorrect")

        shannon_entropy.recommended_counter = np.random.uniform(0, 100, n_items).astype(np.int)
        self.assertTrue(np.isclose(9.6, shannon_entropy.get_metric_value(), atol=1e-1), "metric incorrect")


    def test_Diversity_list_all_equals(self):

        from RecSysFramework.Evaluation.Metrics import Diversity_MeanInterList
        import scipy.sparse as sps

        n_items = 3
        n_users = 10
        cutoff = min(5, n_items)

        # create recommendation list
        URM_predicted_row = []
        URM_predicted_col = []

        diversity_list = Diversity_MeanInterList(n_items, cutoff)
        item_id_list = np.arange(0, n_items, dtype=np.int)

        for n_user in range(n_users):

            np.random.shuffle(item_id_list)
            recommended = item_id_list[:cutoff]
            URM_predicted_row.extend([n_user]*cutoff)
            URM_predicted_col.extend(recommended)

            diversity_list.add_recommendations(recommended)

        object_diversity = diversity_list.get_metric_value()

        URM_predicted_data = np.ones_like(URM_predicted_row)

        URM_predicted_sparse = sps.csr_matrix((URM_predicted_data, (URM_predicted_row, URM_predicted_col)), dtype=np.int)

        co_counts = URM_predicted_sparse.dot(URM_predicted_sparse.T).toarray()
        np.fill_diagonal(co_counts, 0)

        all_user_couples_count = n_users**2 - n_users

        diversity_cumulative = 1 - co_counts/cutoff
        np.fill_diagonal(diversity_cumulative, 0)

        diversity_cooccurrence = diversity_cumulative.sum()/all_user_couples_count

        self.assertTrue(np.isclose(diversity_cooccurrence, object_diversity, atol=1e-4), "metric incorrect")


    def test_Diversity_list(self):

        from RecSysFramework.Evaluation.Metrics import Diversity_MeanInterList
        import scipy.sparse as sps

        n_items = 500
        n_users = 1000
        cutoff = 10

        # create recommendation list
        URM_predicted_row = []
        URM_predicted_col = []

        diversity_list = Diversity_MeanInterList(n_items, cutoff)
        item_id_list = np.arange(0, n_items, dtype=np.int)

        for n_user in range(n_users):

            np.random.shuffle(item_id_list)
            recommended = item_id_list[:cutoff]
            URM_predicted_row.extend([n_user]*cutoff)
            URM_predicted_col.extend(recommended)

            diversity_list.add_recommendations(recommended)

        object_diversity = diversity_list.get_metric_value()

        URM_predicted_data = np.ones_like(URM_predicted_row)

        URM_predicted_sparse = sps.csr_matrix((URM_predicted_data, (URM_predicted_row, URM_predicted_col)), dtype=np.int)

        co_counts = URM_predicted_sparse.dot(URM_predicted_sparse.T).toarray()
        np.fill_diagonal(co_counts, 0)

        all_user_couples_count = n_users**2 - n_users

        diversity_cumulative = 1 - co_counts/cutoff
        np.fill_diagonal(diversity_cumulative, 0)

        diversity_cooccurrence = diversity_cumulative.sum()/all_user_couples_count

        self.assertTrue(np.isclose(diversity_cooccurrence, object_diversity, atol=1e-4), "metric incorrect")


    def test_AUC(self):

        from RecSysFramework.Evaluation.Metrics import roc_auc, ROC_AUC

        metric = ROC_AUC()

        pos_items = np.asarray([2, 4])
        ranked_list = np.asarray([1, 2, 3, 4, 5])

        is_relevant = np.in1d(ranked_list, pos_items, assume_unique=True)

        metric.add_recommendations(is_relevant, pos_items)

        self.assertTrue(np.allclose(roc_auc(is_relevant),
                                    (2. / 3 + 1. / 3) / 2))

        self.assertTrue(np.allclose(metric.get_metric_value(), (2. / 3 + 1. / 3) / 2))


    def test_Recall(self):

        from RecSysFramework.Evaluation.Metrics import recall, Recall

        metric = Recall()

        pos_items = np.asarray([2, 4, 5, 10])
        ranked_list_1 = np.asarray([1, 2, 3, 4, 5])
        ranked_list_2 = np.asarray([10, 5, 2, 4, 3])
        ranked_list_3 = np.asarray([1, 3, 6, 7, 8])

        is_relevant = np.in1d(ranked_list_1, pos_items, assume_unique=True)
        metric.add_recommendations(is_relevant, pos_items)
        self.assertTrue(np.allclose(metric.get_metric_value_per_user()[0], 3. / 4))
        self.assertTrue(np.allclose(recall(is_relevant, pos_items), 3. / 4))

        is_relevant = np.in1d(ranked_list_2, pos_items, assume_unique=True)
        metric.add_recommendations(is_relevant, pos_items)
        self.assertTrue(np.allclose(metric.get_metric_value_per_user()[1], 1.0))
        self.assertTrue(np.allclose(recall(is_relevant, pos_items), 1.0))

        is_relevant = np.in1d(ranked_list_3, pos_items, assume_unique=True)
        metric.add_recommendations(is_relevant, pos_items)
        self.assertTrue(np.allclose(metric.get_metric_value_per_user()[2], 0.0))
        self.assertTrue(np.allclose(recall(is_relevant, pos_items), 0.0))

        self.assertTrue(np.allclose(metric.get_metric_value(), np.mean([0, 1.0, 3/4])))


    def test_Precision(self):

        from RecSysFramework.Evaluation.Metrics import precision, Precision

        metric = Precision()

        pos_items = np.asarray([2, 4, 5, 10])
        ranked_list_1 = np.asarray([1, 2, 3, 4, 5])
        ranked_list_2 = np.asarray([10, 5, 2, 4, 3])
        ranked_list_3 = np.asarray([1, 3, 6, 7, 8])

        is_relevant = np.in1d(ranked_list_1, pos_items, assume_unique=True)
        metric.add_recommendations(is_relevant, pos_items)
        self.assertTrue(np.allclose(metric.get_metric_value_per_user()[0], 3. / 5))
        self.assertTrue(np.allclose(precision(is_relevant), 3. / 5))

        is_relevant = np.in1d(ranked_list_2, pos_items, assume_unique=True)
        metric.add_recommendations(is_relevant, pos_items)
        self.assertTrue(np.allclose(metric.get_metric_value_per_user()[1], 4/5))
        self.assertTrue(np.allclose(precision(is_relevant), 4. / 5))

        is_relevant = np.in1d(ranked_list_3, pos_items, assume_unique=True)
        metric.add_recommendations(is_relevant, pos_items)
        self.assertTrue(np.allclose(metric.get_metric_value_per_user()[2], 0.0))
        self.assertTrue(np.allclose(precision(is_relevant), 0.0))

        self.assertTrue(np.allclose(metric.get_metric_value(), np.mean([0, 4/5, 3/5])))


    def test_RR(self):

        from RecSysFramework.Evaluation.Metrics import rr, MRR

        metric = MRR()

        pos_items = np.asarray([2, 4, 5, 10])
        ranked_list_1 = np.asarray([1, 2, 3, 4, 5])
        ranked_list_2 = np.asarray([10, 5, 2, 4, 3])
        ranked_list_3 = np.asarray([1, 3, 6, 7, 8])

        is_relevant = np.in1d(ranked_list_1, pos_items, assume_unique=True)
        metric.add_recommendations(is_relevant, pos_items)
        self.assertTrue(np.allclose(metric.get_metric_value_per_user()[0], 1./2))
        self.assertTrue(np.allclose(rr(is_relevant), 1. / 2))

        is_relevant = np.in1d(ranked_list_2, pos_items, assume_unique=True)
        metric.add_recommendations(is_relevant, pos_items)
        self.assertTrue(np.allclose(metric.get_metric_value_per_user()[1], 1.))
        self.assertTrue(np.allclose(rr(is_relevant), 1.))

        is_relevant = np.in1d(ranked_list_3, pos_items, assume_unique=True)
        metric.add_recommendations(is_relevant, pos_items)
        self.assertTrue(np.allclose(metric.get_metric_value_per_user()[2], 0.))
        self.assertTrue(np.allclose(rr(is_relevant), 0.0))

        self.assertTrue(np.allclose(metric.get_metric_value(), np.mean([0, 1., 1./2])))


    def test_MAP(self):

        from RecSysFramework.Evaluation.Metrics import MAP

        pos_items = np.asarray([2, 4, 5, 10])
        ranked_list_1 = np.asarray([1, 2, 3, 4, 5])
        ranked_list_2 = np.asarray([10, 5, 2, 4, 3])
        ranked_list_3 = np.asarray([1, 3, 6, 7, 8])
        ranked_list_4 = np.asarray([11, 12, 13, 14, 15, 16, 2, 4, 5, 10])
        ranked_list_5 = np.asarray([2, 11, 12, 13, 14, 15, 4, 5, 10, 16])

        is_relevant = np.in1d(ranked_list_1, pos_items, assume_unique=True)
        map_obj = MAP()
        map_obj.add_recommendations(is_relevant, pos_items)
        self.assertTrue(np.allclose(map_obj.get_metric_value(), (1. / 2 + 2. / 4 + 3. / 5) / 4))

        is_relevant = np.in1d(ranked_list_2, pos_items, assume_unique=True)
        map_obj = MAP()
        map_obj.add_recommendations(is_relevant, pos_items)
        self.assertTrue(np.allclose(map_obj.get_metric_value(), 1.0))

        is_relevant = np.in1d(ranked_list_3, pos_items, assume_unique=True)
        map_obj = MAP()
        map_obj.add_recommendations(is_relevant, pos_items)
        self.assertTrue(np.allclose(map_obj.get_metric_value(), 0.0))

        is_relevant = np.in1d(ranked_list_4, pos_items, assume_unique=True)
        map_obj = MAP()
        map_obj.add_recommendations(is_relevant, pos_items)
        self.assertTrue(np.allclose(map_obj.get_metric_value(), (1. / 7 + 2. / 8 + 3. / 9 + 4. / 10) / 4))

        is_relevant = np.in1d(ranked_list_5, pos_items, assume_unique=True)
        map_obj = MAP()
        map_obj.add_recommendations(is_relevant, pos_items)
        self.assertTrue(np.allclose(map_obj.get_metric_value(), (1. + 2. / 7 + 3. / 8 + 4. / 9) / 4))


    def test_NDCG(self):

        from RecSysFramework.Evaluation.Metrics import dcg, ndcg, NDCG

        metric = NDCG()

        pos_items = np.asarray([2, 4, 5, 10])
        pos_relevances = np.asarray([5, 4, 3, 2])
        ranked_list_1 = np.asarray([1, 2, 3, 4, 5])  # rel = 0, 5, 0, 4, 3
        ranked_list_2 = np.asarray([10, 5, 2, 4, 3])  # rel = 2, 3, 5, 4, 0
        ranked_list_3 = np.asarray([1, 3, 6, 7, 8])  # rel = 0, 0, 0, 0, 0
        idcg = ((2 ** 5 - 1) / np.log(2) +
                (2 ** 4 - 1) / np.log(3) +
                (2 ** 3 - 1) / np.log(4) +
                (2 ** 2 - 1) / np.log(5))
        ndcg_vals = [
            ((2 ** 5 - 1) / np.log(3) +
             (2 ** 4 - 1) / np.log(5) +
             (2 ** 3 - 1) / np.log(6)) / idcg,
            ((2 ** 2 - 1) / np.log(2) +
             (2 ** 3 - 1) / np.log(3) +
             (2 ** 5 - 1) / np.log(4) +
             (2 ** 4 - 1) / np.log(5)) / idcg,
            0.
        ]
        self.assertTrue(np.allclose(dcg(np.sort(pos_relevances)[::-1]), idcg))
        self.assertTrue(np.allclose(ndcg(ranked_list_1, pos_items, pos_relevances),
                                    ndcg_vals[0]))
        self.assertTrue(np.allclose(ndcg(ranked_list_2, pos_items, pos_relevances),
                                    ndcg_vals[1]))
        self.assertTrue(np.allclose(ndcg(ranked_list_3, pos_items, pos_relevances), ndcg_vals[2]))
        metric.add_recommendations(ranked_list_1, pos_items, pos_relevances)
        self.assertTrue(np.allclose(metric.get_metric_value_per_user()[0], ndcg_vals[0]))
        metric.add_recommendations(ranked_list_2, pos_items, pos_relevances)
        self.assertTrue(np.allclose(metric.get_metric_value_per_user()[1], ndcg_vals[1]))
        metric.add_recommendations(ranked_list_3, pos_items, pos_relevances)
        self.assertTrue(np.allclose(metric.get_metric_value_per_user()[2], ndcg_vals[2]))

        self.assertTrue(np.allclose(metric.get_metric_value(), np.mean(ndcg_vals)))



if __name__ == '__main__':
    unittest.main()

