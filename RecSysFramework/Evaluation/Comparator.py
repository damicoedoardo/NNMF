'''
Created on Wed Sep 18 2019

@author XXX
'''

import numpy as np
import scipy.sparse as sps
import time
import sys
import copy
from tqdm import tqdm
from enum import Enum
from RecSysFramework.Utils import seconds_to_biggest_unit
from RecSysFramework.Utils.compute_popularity import compute_popularity_user
from RecSysFramework.Utils.WriteTextualFile import WriteTextualFile
from RecSysFramework.Utils.Log import dict_to_string
from scipy.stats import spearmanr
import rbo


class Comparator(object):
    """ Abstract comprator

        Used in general for statistics that take into account more recommenders jointly.
        eg: variety of recommendations, it measures the difference between items recommended.
    """

    COMPARATOR_NAME = "Comparator"
    metrics = ['jaccard', 'spearman', 'RBO']

    def __init__(self, URM_test, recommenders_to_compare, cutoff=5, metrics_list=['jaccard', 'spearman', 'RBO'],
                 minRatingsPerUser=1, exclude_seen=True, exclude_top_pop=False, save_to_file=True, verbose=True):
        """

        Arguments:
            recommenders_to_compare {RecommenderBase} -- instances to be compared
            URM_test {[type]} -- ground thruth

        Keyword Arguments:
            cutoff_list {list} -- cutoff for the list of recommendations to consider
            metrics_list {list} -- list of metrics to be computed (default: [])
            minRatingsPerUser {int} -- recommendations are provided only for users with at least this values of ratings (default: {1})
            exclude_seen {bool} -- shall we exlude from the recommendations the items seen at training time? (default: {True})
            exclude_top_pop {bool} -- shall we exlude from the recommendations the most popular items? (default: {False})
            save_to_file {bool} -- if True, the results are saved into a file, otherwise just printed
        """

        super(Comparator, self).__init__()

        self.minRatingsPerUser = minRatingsPerUser
        self.exclude_seen = exclude_seen
        self.metrics_list = metrics_list
        self.exclude_top_pop = exclude_top_pop
        self.recommenders_to_compare = recommenders_to_compare
        self.cutoff = cutoff
        self.save_to_file = save_to_file
        self.verbose = verbose

        if save_to_file:
            recs = ''
            for r in recommenders_to_compare:
                recs += r.RECOMMENDER_NAME + '_'
            self.file = WriteTextualFile(
                'ComparationResults', recs, append_datetime=True)

        if URM_test is not None:
            self.URM_test = sps.csr_matrix(URM_test)
            self.n_users, self.n_items = URM_test.shape

        numRatings = np.ediff1d(self.URM_test.tocsr().indptr)
        self.usersToEvaluate = np.arange(
            self.n_users)[numRatings >= self.minRatingsPerUser]

    def _evaluate_recommender(self, recommender_object):
        """ should return
            [{user_x: [rec_items_for_x], user_y: [rec_items_for_y]},
             {user_i: [rec_items_for_i], user_j: [rec_items_for_j]},
             ...
            ],
            [descr1, 
            descr2, 
            ...
            ]

           A subclass will give an implementation to this method
        """
        pass

    def _print(self, text, only_on_file=False):
        if self.verbose:
            print(text)
        if self.save_to_file:
            self.file.write_line(text)

    def compare(self):
        """compares the recommenders according to the metrics defined

        Returns:
            [string] -- string describing the results
            [dict] -- the format is {'descr': descr, 'cutoff': cutoff, 'value': value}
        """
        return_string = ''
        return_dict = {}
        for m in self.metrics_list:
            assert m in self.metrics, "metric provided should be among {}".format(
                self.metrics)

            return_string += 'cutoff: {}\n'.format(str(self.cutoff))
            return_string += 'computing {}. considering only the first two recommenders provided\n'.format(
                m)

            if hasattr(self.recommenders_to_compare[0], 'model_parameters') and hasattr(self.recommenders_to_compare[1], 'model_parameters'):
                return_string += 'recommender 1: {}\n\n'.format(dict_to_string(
                    self.recommenders_to_compare[0].model_parameters, style='constructor'))
                return_string += 'recommender 2: {}\n\n'.format(dict_to_string(
                    self.recommenders_to_compare[1].model_parameters, style='constructor'))

            evaluation_results_first, descr = self._evaluate_recommender(
                self.recommenders_to_compare[0])
            evaluation_results_second, _ = self._evaluate_recommender(
                self.recommenders_to_compare[1])

            for idx in range(len(evaluation_results_second)):

                if m == 'jaccard':
                    result = self.compute_jaccard(
                        evaluation_results_first[idx], evaluation_results_second[idx])
                elif m == 'RBO':
                    result = self.compute_RBO(
                        evaluation_results_first[idx], evaluation_results_second[idx])

                return_string += '{} on {}: {}\n'.format(
                    m, descr[idx], result)
                return_dict['{}_{}_{}'.format(
                    m, descr[idx], self.cutoff)] = result

            self._print(return_string)
        return return_string, return_dict

    def compute_jaccard(self, d1, d2):
        j = []
        for key, l1 in d1.items():
            s1 = set(l1)
            s2 = set(d2[key])
            j.append(len(s1 & s2)/len(s1 | s2))
        return sum(j)/len(j)

    def compute_RBO(self, d1, d2):
        j = []
        for key, l1 in d1.items():
            l2 = d2[key]
            j.append(rbo.RankingSimilarity(l1, l2).rbo())
        return sum(j)/len(j)


class ComparatorHoldout(Comparator):
    """ComparatorHoldout"""

    EVALUATOR_NAME = "ComparatorHoldout"

    def _evaluate_recommender(self, recommender_object):
        recommended_items_batch_list, _ = recommender_object.recommend(self.usersToEvaluate,
                                                                       remove_seen_flag=self.exclude_seen,
                                                                       cutoff=self.cutoff,
                                                                       remove_top_pop_flag=self.exclude_top_pop,
                                                                       return_scores=True)
        return [dict(zip(self.usersToEvaluate, recommended_items_batch_list))], ['all_users']


class ComparatorHoldoutUserPopularity(Comparator):
    """ComparatorHoldoutUserPopularity

        evaluates the recommender considering different sets of users based on their popularity
    """

    EVALUATOR_NAME = "ComparatorHoldoutUserPopularity"

    def __init__(self, URM_train, *pos_args, cuts=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1], **key_args):
        self.URM_train = URM_train
        self.cuts = cuts
        super(ComparatorHoldoutUserPopularity,
              self).__init__(*pos_args, **key_args)

    def _recommend_in_batch(self, recommender_object, users, remove_seen_flag, cutoff, remove_top_pop_flag):
        r = []
        size = 1000
        n_users = len(users)

        n_batch = n_users // size
        for idx in range(n_batch):
            r += recommender_object.recommend(
                users[size*idx: size*(idx+1)], remove_seen_flag=remove_seen_flag, cutoff=cutoff, remove_top_pop_flag=remove_top_pop_flag, return_scores=True)[0]
        r += recommender_object.recommend(
            users[(size*n_batch) % n_users: n_users], remove_seen_flag=remove_seen_flag, cutoff=cutoff, remove_top_pop_flag=remove_top_pop_flag, return_scores=True)[0]
        return r

    def _evaluate_recommender(self, recommender_object):
        pop = compute_popularity_user(self.URM_train, ordered=True)
        r = []
        descr = []

        users, interactions = zip(*pop)
        users = np.array(users)
        interactions = np.array(interactions)
        cum_sum_interactions = np.cumsum(interactions)
        tot_interactions = np.sum(interactions)
        recommended_items_all_users = self._recommend_in_batch(recommender_object, 
                                                               np.sort(users),
                                                               self.exclude_seen,
                                                               self.cutoff,
                                                               self.exclude_top_pop,
                                                               )
        recommended_items_all_users = np.array(recommended_items_all_users)

        for cut in self.cuts:
            users_in_cut = users[cum_sum_interactions < cut*tot_interactions]
            recommended_items_batch_list = recommended_items_all_users[users_in_cut, :].tolist()
            r.append(dict(zip(users_in_cut, recommended_items_batch_list)))
            descr.append('{}'.format(cut))

        return r, descr
