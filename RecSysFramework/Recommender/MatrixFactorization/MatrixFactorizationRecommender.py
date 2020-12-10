#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 16/09/2017

@author: XXX
"""

from RecSysFramework.Recommender import Recommender
from RecSysFramework.Recommender.KNN import ItemKNNCustomSimilarity
from RecSysFramework.Utils import check_matrix
from RecSysFramework.Utils import EarlyStoppingModel
from RecSysFramework.Recommender.DataIO import DataIO
import seaborn as sns
from sklearn.preprocessing import normalize
import numpy as np
import os
import scipy.sparse as sps
from RecSysFramework.Utils.compute_popularity import compute_popularity_item, compute_popularity_user
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math

def compute_W_sparse_from_item_latent_factors(ITEM_factors, topK=100):

    n_items, n_factors = ITEM_factors.shape

    block_size = 100

    start_item = 0
    end_item = 0

    values = []
    rows = []
    cols = []

    # Compute all similarities for each item using vectorization
    while start_item < n_items:

        end_item = min(n_items, start_item + block_size)

        this_block_weight = np.dot(ITEM_factors[start_item:end_item, :], ITEM_factors.T)


        for col_index_in_block in range(this_block_weight.shape[0]):

            this_column_weights = this_block_weight[col_index_in_block, :]
            item_original_index = start_item + col_index_in_block

            # Sort indices and select TopK
            # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
            # - Partition the data to extract the set of relevant items
            # - Sort only the relevant items
            # - Get the original item index
            relevant_items_partition = (-this_column_weights).argpartition(topK-1)[0:topK]
            relevant_items_partition_sorting = np.argsort(-this_column_weights[relevant_items_partition])
            top_k_idx = relevant_items_partition[relevant_items_partition_sorting]

            # Incrementally build sparse matrix, do not add zeros
            notZerosMask = this_column_weights[top_k_idx] != 0.0
            numNotZeros = np.sum(notZerosMask)

            values.extend(this_column_weights[top_k_idx][notZerosMask])
            rows.extend(top_k_idx[notZerosMask])
            cols.extend(np.ones(numNotZeros) * item_original_index)

        start_item += block_size

    W_sparse = sps.csr_matrix((values, (rows, cols)),
                              shape=(n_items, n_items),
                              dtype=np.float32)

    return W_sparse



class BaseMatrixFactorizationRecommender(Recommender):
    """
    This class refers to a BaseRecommender KNN which uses matrix factorization,
    it provides functions to compute item's score as well as a function to save the W_matrix

    The prediction for cold users will always be -inf for ALL items
    """

    def __init__(self, URM_train):
        super(BaseMatrixFactorizationRecommender, self).__init__(URM_train)

        self.use_bias = False

        self.user_update_count = None
        self.item_update_count = None

        self._cold_user_KNN_model_flag = False
        self._cold_user_KNN_estimated_factors_flag = False
        self._warm_user_KNN_mask = np.zeros(len(self._get_cold_user_mask()), dtype=np.bool)


    def set_URM_train(self, URM_train_new, estimate_model_for_cold_users = False, topK = 100, **kwargs):
        """

        :param URM_train_new:
        :param estimate_item_similarity_for_cold_users: Set to TRUE if you want to estimate the item-item similarity for cold users to be used as in a KNN algorithm
        :param topK: 100
        :param kwargs:
        :return:
        """

        assert self.URM_train.shape == URM_train_new.shape, "{}: set_URM_train old and new URM train have different shapes".format(self.RECOMMENDER_NAME)

        if len(kwargs)>0:
            self._print("set_URM_train keyword arguments not supported for this recommender class. Received: {}".format(kwargs))

        URM_train_new = check_matrix(URM_train_new, 'csr', dtype=np.float32)
        profile_length_new = np.ediff1d(URM_train_new.indptr)

        if estimate_model_for_cold_users == "itemKNN":

            self._print("Estimating ItemKNN model from ITEM latent factors...")

            W_sparse = compute_W_sparse_from_item_latent_factors(self.ITEM_factors, topK=topK)

            self._ItemKNNRecommender = ItemKNNCustomSimilarity(URM_train_new)
            self._ItemKNNRecommender.fit(W_sparse, topK=topK)
            self._ItemKNNRecommender_topK = topK

            self._cold_user_KNN_model_flag = True
            self._warm_user_KNN_mask = profile_length_new > 0

            self._print("Estimating ItemKNN model from ITEM latent factors... done!")

        elif estimate_model_for_cold_users == "mean_item_factors":

            self._print("Estimating USER latent factors from ITEM latent factors...")

            cold_user_mask_previous = self._get_cold_user_mask()
            profile_length_sqrt = np.sqrt(profile_length_new)

            self.USER_factors[cold_user_mask_previous,:] = URM_train_new.dot(self.ITEM_factors)[cold_user_mask_previous,:]
            self._cold_user_KNN_estimated_factors_flag = True

            #Divide every row for the sqrt of the profile length
            for user_index in range(self.n_users):
                if cold_user_mask_previous[user_index] and profile_length_sqrt[user_index] > 0:

                    self.USER_factors[user_index, :] /= profile_length_sqrt[user_index]

            self._print("Estimating USER latent factors from ITEM latent factors... done!")

        self.URM_train = check_matrix(URM_train_new.copy(), 'csr', dtype=np.float32)
        self.URM_train.eliminate_zeros()


    #########################################################################################################
    ##########                                                                                     ##########
    ##########                               COMPUTE ITEM SCORES                                   ##########
    ##########                                                                                     ##########
    #########################################################################################################


    def _compute_item_score(self, user_id_array, items_to_compute = None):
        """
        USER_factors is n_users x n_factors
        ITEM_factors is n_items x n_factors

        The prediction for cold users will always be -inf for ALL items

        :param user_id_array:
        :param items_to_compute:
        :return:
        """

        assert self.USER_factors.shape[1] == self.ITEM_factors.shape[1], \
            "{}: User and Item factors have inconsistent shape".format(self.RECOMMENDER_NAME)

        assert self.USER_factors.shape[0] > user_id_array.max(),\
                "{}: Cold users not allowed. Users in trained model are {}, requested prediction for users up to {}".format(
                self.RECOMMENDER_NAME, self.USER_factors.shape[0], user_id_array.max())

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.ITEM_factors.shape[0]), dtype=np.float32)*np.inf
            item_scores[:, items_to_compute] = np.dot(self.USER_factors[user_id_array], self.ITEM_factors[items_to_compute,:].T)

        else:
            item_scores = np.dot(self.USER_factors[user_id_array], self.ITEM_factors.T)

        # No need to select only the specific negative items or warm users because the -inf score will not change
        if self.use_bias:
            item_scores += self.ITEM_bias + self.GLOBAL_bias
            item_scores = (item_scores.T + self.USER_bias[user_id_array]).T

        item_scores = self._compute_item_score_postprocess_for_cold_users(user_id_array, item_scores, items_to_compute = items_to_compute)
        item_scores = self._compute_item_score_postprocess_for_cold_items(item_scores)

        return item_scores


    def _compute_item_score_postprocess_for_cold_users(self, user_id_array, item_scores, items_to_compute = None):
        """
        Remove cold users from the computed item scores, setting them to -inf
        Or estimate user factors with specified method
        :param user_id_array:
        :param item_scores:
        :return:
        """

        cold_users_batch_mask = self._get_cold_user_mask()[user_id_array]

        # Set as -inf all cold user scores
        if cold_users_batch_mask.any() and not self._cold_user_KNN_estimated_factors_flag:

            if self._cold_user_KNN_model_flag:
                # Add KNN scores for users cold for MF but warm in KNN model
                cold_users_in_MF_warm_in_KNN_mask = np.logical_and(cold_users_batch_mask, self._warm_user_KNN_mask[user_id_array])

                item_scores[cold_users_in_MF_warm_in_KNN_mask, :] = self._ItemKNNRecommender._compute_item_score(user_id_array[cold_users_in_MF_warm_in_KNN_mask], items_to_compute=items_to_compute)

                # Set cold users as those neither in MF nor in KNN
                cold_users_batch_mask = np.logical_and(cold_users_batch_mask, np.logical_not(cold_users_in_MF_warm_in_KNN_mask))

            # Set as -inf all remaining cold user scores
            item_scores[cold_users_batch_mask, :] = - np.ones_like(item_scores[cold_users_batch_mask, :]) * np.inf

        return item_scores


    #########################################################################################################
    ##########                                                                                     ##########
    ##########                                LOAD AND SAVE                                        ##########
    ##########                                                                                     ##########
    #########################################################################################################


    def _get_dict_to_save(self):

        data_dict_to_save = {"USER_factors": self.USER_factors,
                              "ITEM_factors": self.ITEM_factors,
                              "use_bias": self.use_bias,
                              "_cold_user_mask": self._cold_user_mask,
                              "_cold_user_KNN_model_flag": self._cold_user_KNN_model_flag,
                              "_cold_user_KNN_estimated_factors_flag": self._cold_user_KNN_estimated_factors_flag}

        if self.use_bias:
            data_dict_to_save["ITEM_bias"] = self.ITEM_bias
            data_dict_to_save["USER_bias"] = self.USER_bias
            data_dict_to_save["GLOBAL_bias"] = self.GLOBAL_bias

        if self._cold_user_KNN_model_flag:
            data_dict_to_save["_ItemKNNRecommender_W_sparse"] = self._ItemKNNRecommender.W_sparse
            data_dict_to_save["_ItemKNNRecommender_topK"] = self._ItemKNNRecommender_topK

        return data_dict_to_save


    def load_model(self, folder_path, file_name=None):
        super(BaseMatrixFactorizationRecommender, self).load_model(folder_path, file_name=file_name)

        if self._cold_user_KNN_model_flag:
            self._ItemKNNRecommender = ItemKNNCustomSimilarity(self.URM_train)
            self._ItemKNNRecommender.fit(self._ItemKNNRecommender_W_sparse, topK=self._ItemKNNRecommender_topK)

            del self._ItemKNNRecommender_W_sparse
            del self._ItemKNNRecommender_topK


    def plot_items_sampled_stats(self, items_to_plot=5000, plot_complete_graphic=True, normalized=False):
        """
        it makes and display how many times a given item has been updated, associating to each of them a color based on
        the popularity

        :param items_to_plot: how many items per plot
        :param plot_complete_graphic: where to plot the complete graphic all at once
        :return:
        """
        assert self.item_update_count is not None, 'the model has not implemented this function yet or have not been trained'

        popularity_list = compute_popularity_item(self.URM_train)
        item, interaction = zip(* popularity_list)
        colors = cm.coolwarm(np.array(interaction))
        color_mapping_dict = dict(zip(item, colors))

        #item, num_sampled = zip(*self.item_update_count)

        unsorted_list = list(self.item_update_count.items())
        sorted_list = sorted(unsorted_list, key=lambda x: x[1])

        item_id, sampled_count = zip(*sorted_list)

        if normalized:
            sampled_count = sampled_count/max(sampled_count)

        # map the popularity of the item to its color
        plot_colors = []
        for id in item_id:
            plot_colors.append(color_mapping_dict[id])

        x_pos = np.arange(len(item_id))

        for i in range(math.ceil(self.n_items/items_to_plot)):
            if (i+1)*items_to_plot < len(item_id):
                x_pos_slice = x_pos[items_to_plot*i:items_to_plot*(i+1)]
                sampled_count_slice = sampled_count[items_to_plot*i:items_to_plot*(i+1)]
                plot_colors_slice = plot_colors[items_to_plot*i:items_to_plot*(i+1)]
            else:
                x_pos_slice = x_pos[items_to_plot*i:-1]
                sampled_count_slice = sampled_count[items_to_plot*i:-1]
                plot_colors_slice = plot_colors[items_to_plot*i:-1]

            plt.bar(x_pos_slice, sampled_count_slice, align='center', color=np.array(plot_colors_slice))
            plt.show()

        if plot_complete_graphic:
            plt.bar(x_pos, sampled_count, align='center', color=np.array(plot_colors))
            plt.show()

    def plot_users_sampled_stats(self):
        #TODO MERGE THIS IN THE METHOD ABOVE!!!
        assert self.user_update_count is not None, 'the model has not implemented this function yet or have not been trained'

        popularity_list = compute_popularity_user(self.URM_train)
        user, interaction = zip(* popularity_list)
        colors = cm.PiYG(np.array(interaction))
        color_mapping_dict = dict(zip(user, colors))

        #item, num_sampled = zip(*self.item_update_count)

        unsorted_list = list(self.user_update_count.items())
        sorted_list = sorted(unsorted_list, key=lambda x: x[1])

        user_id, sampled_count = zip(*sorted_list)

        sampled_count = sampled_count/max(sampled_count)

        # map the popularity of the item to its color
        plot_colors = []
        for id in user_id:
            plot_colors.append(color_mapping_dict[id])

        x_pos = np.arange(len(user_id))
        plt.bar(x_pos, sampled_count, align='center', color=np.array(plot_colors))
        plt.show()

    def plot_latent_representations_heatmap(self):
        """
        plot the latent representation of the items and user using an heatmap, the value of the representation
        are normalized
        """
        # retrieve the latent factors of the users and the items
        print('Normalizing...')
        items_factors_normalized = normalize(self.ITEM_factors, axis=1, norm='l1')
        users_factors_normalized = normalize(self.USER_factors, axis=1, norm ='l1')
        print('Done!')

        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)

        heat_map_items = sns.heatmap(items_factors_normalized, xticklabels=False, yticklabels=False, annot=False,
                               cmap='Reds', ax=ax1)
        heat_map_users = sns.heatmap(users_factors_normalized, xticklabels=False, yticklabels=False, annot=False,
                               cmap='Greens', ax=ax2)
        plt.show()



