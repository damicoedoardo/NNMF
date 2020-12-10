#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 07/09/17

@author: XXX, XXX, XXX
"""

import sys
import numpy as np
from tqdm import tqdm
import sklearn.preprocessing as prep
from RecSysFramework.Recommender.MatrixFactorization import BaseMatrixFactorizationRecommender
from RecSysFramework.Utils.EarlyStopping import EarlyStoppingModel
from RecSysFramework.Utils import check_matrix
import similaripy as sim
import scipy.sparse as sps
from functools import partial
from RecSysFramework.Utils.compute_popularity import compute_popularity_item
from tkinter.filedialog import askopenfilename



class _MatrixFactorization_Cython(BaseMatrixFactorizationRecommender, EarlyStoppingModel):
    RECOMMENDER_NAME = "MatrixFactorization_Cython_Recommender"

    def __init__(self, URM_train, algorithm_name="MF_BPR"):
        super(_MatrixFactorization_Cython, self).__init__(URM_train)

        self.normalize = False
        self.algorithm_name = algorithm_name

    def fit(self, epochs=300, batch_size=1000,
            num_factors=10, positive_threshold_BPR=None,
            learning_rate=0.001, use_bias=True,
            sgd_mode='adam',
            negative_interactions_quota=0.5,
            dropout_quota=None,
            latent_factors_initialization={'type':'RANDOM', 'range':[0, 0.1]},
            user_reg=0.0, item_reg=0.0, bias_reg=0.0, positive_reg=0.0, negative_reg=0.0,
            verbose=False, random_seed=-1, save_fitted_model=False,
            **earlystopping_kwargs):
        """
        :param latent_factors_initialization: has to be a DICTIONARY with one of the following configuration:
        NORMAL distributed latent factors initialization -> {type:'NORMAL', 'mean':float, 'std':float}
        RANDOM distributed latent factors initialization {type:'RANDOM', 'range':[int start, int end]}

        """

        self.num_factors = num_factors
        self.use_bias = use_bias
        self.sgd_mode = sgd_mode
        self.verbose = verbose
        self.positive_threshold_BPR = positive_threshold_BPR
        self.learning_rate = learning_rate

        assert negative_interactions_quota >= 0.0 and negative_interactions_quota < 1.0, \
            "{}: negative_interactions_quota must be a float value >=0 and < 1.0, provided was '{}'" \
                .format(self.RECOMMENDER_NAME, negative_interactions_quota)
        self.negative_interactions_quota = negative_interactions_quota

        # Import compiled module
        from RecSysFramework.Recommender.MatrixFactorization.Cython.MatrixFactorization_Cython_Epoch import MatrixFactorization_Cython_Epoch
        from RecSysFramework.Recommender.MatrixFactorization.Cython.old_MatrixFactorization_Cython_Epoch import NNMF_BPR_Cython_Epoch

        if self.algorithm_name in ["FUNK_SVD", "ASY_SVD"]:

            self.cythonEpoch = MatrixFactorization_Cython_Epoch(self.URM_train,
                                                                algorithm_name=self.algorithm_name,
                                                                n_factors=self.num_factors,
                                                                learning_rate=learning_rate,
                                                                sgd_mode=sgd_mode,
                                                                user_reg=user_reg,
                                                                item_reg=item_reg,
                                                                bias_reg=bias_reg,
                                                                batch_size=batch_size,
                                                                use_bias=use_bias,
                                                                latent_factors_initialization=latent_factors_initialization,
                                                                negative_interactions_quota=negative_interactions_quota,
                                                                dropout_quota=dropout_quota,
                                                                verbose=verbose,
                                                                random_seed=random_seed)

        elif self.algorithm_name in ["MF_BPR", "BPR_NNMF", "FUNK_NNMF", "PROB_NNMF"]:

            # Select only positive interactions
            URM_train_positive = self.URM_train.copy()

            if self.positive_threshold_BPR is not None:
                URM_train_positive.data = URM_train_positive.data >= self.positive_threshold_BPR
                URM_train_positive.eliminate_zeros()

                assert URM_train_positive.nnz > 0, \
                    "MatrixFactorization_Cython: URM_train_positive is empty, positive threshold is too high"

            self.cythonEpoch = MatrixFactorization_Cython_Epoch(URM_train_positive,
                                                                algorithm_name=self.algorithm_name,
                                                                n_factors=self.num_factors,
                                                                learning_rate=learning_rate,
                                                                sgd_mode=sgd_mode,
                                                                user_reg=user_reg,
                                                                positive_reg=positive_reg,
                                                                negative_reg=negative_reg,
                                                                batch_size=batch_size,
                                                                use_bias=use_bias,
                                                                latent_factors_initialization=latent_factors_initialization,
                                                                dropout_quota=dropout_quota,
                                                                negative_interactions_quota=negative_interactions_quota,
                                                                verbose=verbose,
                                                                random_seed=random_seed)

        self._prepare_model_for_validation()
        self._update_best_model()

        self._train_with_early_stopping(epochs,
                                        algorithm_name=self.algorithm_name,
                                        **earlystopping_kwargs)

        self.USER_factors = self.USER_factors_best
        self.ITEM_factors = self.ITEM_factors_best

        # save user and item update count in self
        user_update_count = self.cythonEpoch.get_user_update_count()
        item_update_count = self.cythonEpoch.get_item_update_count()

        self.user_update_count = {i: user_update_count[i] for i in range(len(user_update_count))}
        self.item_update_count = {i: item_update_count[i] for i in range(len(item_update_count))}

        if self.use_bias:
            self.USER_bias = self.USER_bias_best
            self.ITEM_bias = self.ITEM_bias_best
            self.GLOBAL_bias = self.GLOBAL_bias_best

        if save_fitted_model:
            self.save_model('SingleRunsSavedModels/')

        sys.stdout.flush()

    def _prepare_model_for_validation(self):
        self.USER_factors = self.cythonEpoch.get_USER_factors()
        self.ITEM_factors = self.cythonEpoch.get_ITEM_factors()

        if self.use_bias:
            self.USER_bias = self.cythonEpoch.get_USER_bias()
            self.ITEM_bias = self.cythonEpoch.get_ITEM_bias()
            self.GLOBAL_bias = self.cythonEpoch.get_GLOBAL_bias()

    def _update_best_model(self):
        self.USER_factors_best = self.USER_factors.copy()
        self.ITEM_factors_best = self.ITEM_factors.copy()

        if self.use_bias:
            self.USER_bias_best = self.USER_bias.copy()
            self.ITEM_bias_best = self.ITEM_bias.copy()
            self.GLOBAL_bias_best = self.GLOBAL_bias

    def _run_epoch(self, num_epoch):
        self.cythonEpoch.epochIteration_Cython()



class NNMF(_MatrixFactorization_Cython):

    def __init__(self, *pos_args, **key_args):

        self.ICM = None

        # if we pass the icm as parameters means we want use a content similarity
        if 'icm' in key_args:
            self.ICM = key_args['icm'].tocsr()
            key_args.pop('icm')        

        super(NNMF, self).__init__(*pos_args, **key_args)

    def load_model(self, folder_path='', file_name=None, gui=False):
        """
        override the method to use a gui for select the filename
        :return:
        """
        if gui:
            file_name = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
            if 'EvaluationResults/' in file_name:
                file_name = file_name.split('EvaluationResults/')[1]
                folder_path = 'EvaluationResults/'
            elif 'SingleRunsSavedModels/' in file_name:
                file_name = file_name.split('SingleRunsSavedModels/')[1]
                folder_path = 'SingleRunsSavedModels/'
            elif 'BestModels/' in file_name:
                file_name = file_name.split('BestModels/')[1]
                folder_path = 'BestModels/'
            else:
                raise ValueError('I expect the model to load to be either in EvaluationResults/ '
                                 'or in SingleRunsSavedModels/ or BestModels/')
            file_name = file_name.split('.zip')[0]
        super(NNMF, self).load_model(folder_path=folder_path, file_name=file_name)

    def _compute_popularity_driven_probability_item_sampling(self):
        _, pop_items_list = zip(*compute_popularity_item(self.URM_train, ordered=False))
        pop_items = np.array(pop_items_list, dtype=np.float64)
        pop_items[pop_items>=1] = 1/(pop_items[pop_items>=1]/max(pop_items))
        pop_items = np.array(pop_items)
        return pop_items


    def fit(self, threshold=0, item_k=100, user_k=100, item_shrink=10, user_shrink=10,
            item_eye=False, user_eye=False, normalization=True, **key_args):

        self.model_parameters = locals()

        key_args["use_bias"] = False

        # setting k has an object attribute has to be passed to the run epoch method
        self.negative_interactions_quota=0.5
        self.item_k = item_k
        self.user_k = user_k
        self.user_shrink = user_shrink
        self.item_shrink = item_shrink
        self.user_eye = user_eye
        self.item_eye = item_eye

        # initialize the two similarity matrices as field of the object
        # they will be stored in CSR format
        if user_eye:
            print('USER: Identity matrix for similarity')
            self.user_similarity = sps.eye(self.n_users).tocsr()
        else:
            self.user_similarity = sim.cosine(self.URM_train, k=user_k, shrink=user_shrink, threshold=threshold).tocsr().astype(np.float64)

        if item_eye:
            print('ITEMS: Identity matrix for similarity')
            self.item_similarity = sps.eye(self.n_items).tocsr()
        else:
            if self.ICM is not None:
                print('ITEMS: Content similarity for items')
                self.item_similarity = sim.cosine(self.ICM, k=item_k, shrink=item_shrink, threshold=threshold).tocsr().astype(np.float64)
            else:
                print('ITEMS: Collaborative similarity')
                self.item_similarity = sim.cosine(self.URM_train.T, k=item_k, shrink=item_shrink, threshold=threshold).tocsr().astype(
                    np.float64)

        # normalize by row the similarity matrices
        if normalization:
            self.user_similarity = prep.normalize(self.user_similarity, norm='max', axis=1, copy=False)
            self.item_similarity = prep.normalize(self.item_similarity, norm='max', axis=1, copy=False)


        #self._fix_item_similarity(k)
        self._initialize_similarity_matrices(item_k, user_k)

        super(NNMF, self).fit(**key_args)

    def _fix_item_similarity(self, k):
        assert self.item_similarity.data.max()<1.1
        self.item_similarity.data = np.clip(self.item_similarity.data, self.item_similarity.data.min(),1)

        print('Fixing item similarity...')
        self.item_similarity.setdiag(2)
        for i in tqdm(range(self.item_similarity.shape[0])):
            if (self.item_similarity.indptr[i+1]-self.item_similarity.indptr[i]) > k:
                index_to_remove = self.item_similarity.data[self.item_similarity.indptr[i]:self.item_similarity.indptr[i+1]].argmin()
                self.item_similarity.data[self.item_similarity.indptr[i]:self.item_similarity.indptr[i + 1]][index_to_remove]=0
        self.item_similarity.setdiag(1)
        self.item_similarity.eliminate_zeros()
        print('Done')

    def _get_dict_to_save(self):

        return {"USER_factors": self.USER_factors,
                "ITEM_factors": self.ITEM_factors,
                "use_bias": self.use_bias,
                "item_similarity": self.item_similarity,
                "user_similarity": self.user_similarity,
                }

    def _compute_item_score(self, user_id_array, items_to_compute = None):
        """
        USER_factors is n_users x n_factors
        ITEM_factors is n_items x n_factors

        The prediction for cold users will always be -inf for ALL items.
        This method is the override in order to allow a modified prediction rule
        in case of NNMF

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
            item_scores[:, items_to_compute] = np.dot(np.dot(self.user_similarity[user_id_array], self.USER_factors[user_id_array]),
                                                      np.dot(self.ITEM_factors.T, self.item_similarity[items_to_compute].T))
        else:
            item_scores = np.dot(self.user_similarity[user_id_array, :].dot(self.USER_factors), self.ITEM_factors.T) * self.item_similarity.T

        # No need to select only the specific negative items or warm users because the -inf score will not change
        if self.use_bias:
            item_scores += self.ITEM_bias + self.GLOBAL_bias
            item_scores = (item_scores.T + self.USER_bias[user_id_array]).T

        item_scores = self._compute_item_score_postprocess_for_cold_users(user_id_array, item_scores, items_to_compute = items_to_compute)
        item_scores = self._compute_item_score_postprocess_for_cold_items(item_scores)

        return item_scores


    def _initialize_similarity_matrices(self, item_k, user_k):
        """
        create and save in a proper format the similarity matrices both the user and item similarity

        """

        def _similarity_decomposition(matrix, k):
            """
            given a matrix (N*M) in csr format it returns two matrices
            1) N*k containing the similarity values
            2) N*k containing the column index to which the similarity is associated with
            
            return padded_data, padded_indices
            """

            def _pad_neighbours(data, k):
                if len(data) != k:
                    data = np.pad(data, (0, k-len(data)), 'constant', constant_values=(0,-1))
                return data

            data = np.split(matrix.data, matrix.indptr[1:-1])
            indices = np.split(matrix.indices, matrix.indptr[1:-1])

            pad_data = np.array(list(map(partial(_pad_neighbours, k=k), data)))
            pad_indices = np.array(list(map(partial(_pad_neighbours, k=k), indices)))

            return pad_data, pad_indices

        data_user_similarity, inidices_user_similarity = _similarity_decomposition(self.user_similarity, user_k)
        data_item_similarity, inidices_item_similarity = _similarity_decomposition(self.item_similarity, item_k)

        self.data_user_similarity = memoryview(data_user_similarity)
        self.inidices_user_similarity = memoryview(inidices_user_similarity)
        self.data_item_similarity = memoryview(data_item_similarity)
        self.inidices_item_similarity = memoryview(inidices_item_similarity)

    def _run_epoch(self, num_epoch):
        self.cythonEpoch.epochIteration_Cython(data_user_similarity=self.data_user_similarity,
                                               inidices_user_similarity=self.inidices_user_similarity,
                                               data_item_similarity=self.data_item_similarity,
                                               indices_item_similarity=self.inidices_item_similarity,
                                               item_k=self.item_k,
                                               user_k=self.user_k
                                              )

class BPR_NNMF(NNMF):
    """
        Neareast neighbours matrix factorization optimized with bpr loss

        variant of the standard matrix factorization taking into account during the update of the latent factors
        also the nearest neighbours of the object we are updating
        link: TO BE INSERTED
    """
    RECOMMENDER_NAME = "BPR_NNMF"

    def __init__(self, *pos_args, **key_args):
        super(BPR_NNMF, self).__init__(*pos_args, algorithm_name=self.RECOMMENDER_NAME, **key_args)


class FUNK_NNMF(NNMF):

    RECOMMENDER_NAME = "FUNK_NNMF"

    def __init__(self, *pos_args, **key_args):
        super(FUNK_NNMF, self).__init__(*pos_args, algorithm_name=self.RECOMMENDER_NAME, **key_args)


class PROB_NNMF(NNMF):
    RECOMMENDER_NAME = "PROB_NNMF"

    def __init__(self, *pos_args, **key_args):
        super(PROB_NNMF, self).__init__(*pos_args, algorithm_name=self.RECOMMENDER_NAME, **key_args)


class BPRMF(_MatrixFactorization_Cython):
    """
    Subclas allowing only for MF BPR
    """

    RECOMMENDER_NAME = "BPRMF"

    def __init__(self, *pos_args, **key_args):
        super(BPRMF, self).__init__(*pos_args, algorithm_name="MF_BPR", **key_args)

    def fit(self, **key_args):
        self.model_parameters = locals()
        key_args["use_bias"] = False
        key_args["negative_interactions_quota"] = 0.0

        super(BPRMF, self).fit(**key_args)

    def load_model(self, folder_path='', file_name=None, gui=False):
        """
        override the method to use a gui for select the filename
        :return:
        """
        if gui:
            file_name = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
            if 'EvaluationResults/' in file_name:
                file_name = file_name.split('EvaluationResults/')[1]
                folder_path = 'EvaluationResults/'
            elif 'SingleRunsSavedModels/' in file_name:
                file_name = file_name.split('SingleRunsSavedModels/')[1]
                folder_path = 'SingleRunsSavedModels/'
            elif 'BestModels/' in file_name:
                file_name = file_name.split('BestModels/')[1]
                folder_path = 'BestModels/'
            else:
                raise ValueError('I expect the model to load to be either in EvaluationResults/ '
                                 'or in SingleRunsSavedModels/ or BestModels/')
            file_name = file_name.split('.zip')[0]
        super(BPRMF, self).load_model(folder_path=folder_path, file_name=file_name)

class FunkSVD(_MatrixFactorization_Cython):
    """
    Subclas allowing only for FunkSVD model

    Reference: http://sifter.org/~simon/journal/20061211.html

    Factorizes the rating matrix R into the dot product of two matrices U and V of latent factors.
    U represent the user latent factors, V the item latent factors.
    The model is learned by solving the following regularized Least-squares objective function with Stochastic Gradient Descent
    \operatornamewithlimits{argmin} \limits_{U,V}\frac{1}{2}||R - UV^T||^2_2 + \frac{\lambda}{2}(||U||^2_F + ||V||^2_F)
    Latent factors are initialized from a Normal distribution with given mean and std.

    """

    RECOMMENDER_NAME = "FunkSVD"

    def __init__(self, *pos_args, **key_args):
        super(FunkSVD, self).__init__(*pos_args, algorithm_name="FUNK_SVD", **key_args)

    def fit(self, **key_args):
        self.model_parameters = locals()
        super(FunkSVD, self).fit(**key_args)

    def load_model(self, folder_path='', file_name=None, gui=False):
        """
        override the method to use a gui for select the filename
        :return:
        """
        if gui:
            file_name = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
            if 'EvaluationResults/' in file_name:
                file_name = file_name.split('EvaluationResults/')[1]
                folder_path = 'EvaluationResults/'
            elif 'SingleRunsSavedModels/' in file_name:
                file_name = file_name.split('SingleRunsSavedModels/')[1]
                folder_path = 'SingleRunsSavedModels/'
            elif 'BestModels/' in file_name:
                file_name = file_name.split('BestModels/')[1]
                folder_path = 'BestModels/'
            else:
                raise ValueError('I expect the model to load to be either in EvaluationResults/ '
                                 'or in SingleRunsSavedModels/ or BestModels/')
            file_name = file_name.split('.zip')[0]
        super(FunkSVD, self).load_model(folder_path=folder_path, file_name=file_name)

class AsySVD(_MatrixFactorization_Cython):
    """
    Subclas allowing only for AsymmetricSVD model

    Reference: Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model (Koren, 2008)

    Factorizes the rating matrix R into two matrices X and Y of latent factors, which both represent item latent features.
    Users are represented by aggregating the latent features in Y of items they have already rated.
    Rating prediction is performed by computing the dot product of this accumulated user profile with the target item's
    latent factor in X.

    The model is learned by solving the following regularized Least-squares objective function with Stochastic Gradient Descent
    \operatornamewithlimits{argmin}\limits_{x*,y*}\frac{1}{2}\sum_{i,j \in R}(r_{ij} - x_j^T \sum_{l \in R(i)} r_{il}y_l)^2 + \frac{\lambda}{2}(\sum_{i}{||x_i||^2} + \sum_{j}{||y_j||^2})
    """

    RECOMMENDER_NAME = "AsySVD"

    def __init__(self, *pos_args, **key_args):
        super(AsySVD, self).__init__(*pos_args, algorithm_name="ASY_SVD", **key_args)

    def fit(self, **key_args):
        self.model_parameters = locals()

        if "batch_size" in key_args and key_args["batch_size"] > 1:
            print("{}: batch_size not supported for this recommender, setting to default value 1.".format(
                self.RECOMMENDER_NAME))

        key_args["batch_size"] = 1

        super(AsySVD, self).fit(**key_args)

    def _prepare_model_for_validation(self):
        """
        AsymmetricSVD Computes two |n_items| x |n_features| matrices of latent factors
        ITEM_factors_Y must be used to estimate user's latent factors via the items they interacted with

        :return:
        """

        self.ITEM_factors_Y = self.cythonEpoch.get_USER_factors()
        self.USER_factors = self._estimate_user_factors(self.ITEM_factors_Y)

        self.ITEM_factors = self.cythonEpoch.get_ITEM_factors()

        if self.use_bias:
            self.USER_bias = self.cythonEpoch.get_USER_bias()
            self.ITEM_bias = self.cythonEpoch.get_ITEM_bias()
            self.GLOBAL_bias = self.cythonEpoch.get_GLOBAL_bias()

    def _update_best_model(self):
        self.USER_factors_best = self.USER_factors.copy()
        self.ITEM_factors_best = self.ITEM_factors.copy()
        self.ITEM_factors_Y_best = self.ITEM_factors_Y.copy()

        if self.use_bias:
            self.USER_bias_best = self.USER_bias.copy()
            self.ITEM_bias_best = self.ITEM_bias.copy()
            self.GLOBAL_bias_best = self.GLOBAL_bias

    def _estimate_user_factors(self, ITEM_factors_Y):

        profile_length = np.ediff1d(self.URM_train.indptr)
        profile_length_sqrt = np.sqrt(profile_length)

        # Estimating the USER_factors using ITEM_factors_Y
        if self.verbose:
            print("{}: Estimating user factors... ".format(self.algorithm_name))

        USER_factors = self.URM_train.dot(ITEM_factors_Y)

        # Divide every row for the sqrt of the profile length
        for user_index in range(self.n_users):

            if profile_length_sqrt[user_index] > 0:
                USER_factors[user_index, :] /= profile_length_sqrt[user_index]

        if self.verbose:
            print("{}: Estimating user factors... done!".format(self.algorithm_name))

        return USER_factors

    def set_URM_train(self, URM_train_new, estimate_item_similarity_for_cold_users=False, **kwargs):
        """

        :param URM_train_new:
        :param estimate_item_similarity_for_cold_users: Set to TRUE if you want to estimate the USER_factors for cold users
        :param kwargs:
        :return:
        """

        assert self.URM_train.shape == URM_train_new.shape, \
            "{}: set_URM_train old and new URM train have different shapes" \
                .format(self.RECOMMENDER_NAME)

        if len(kwargs) > 0:
            self._print("set_URM_train keyword arguments not supported for this recommender class. Received: {}"
                        .format(kwargs))

        self.URM_train = check_matrix(URM_train_new.copy(), 'csr', dtype=np.float32)
        self.URM_train.eliminate_zeros()

        # No need to ever use a knn model
        self._cold_user_KNN_model_available = False
        self._cold_user_mask = np.ediff1d(self.URM_train.indptr) == 0

        if estimate_item_similarity_for_cold_users:
            self._print("Estimating USER_factors for cold users...")

            self.USER_factors = self._estimate_user_factors(self.ITEM_factors_Y_best)

            self._print("Estimating USER_factors for cold users... done!")

class BPRMF_AFM(_MatrixFactorization_Cython):
    """

    Subclass for BPRMF with Attribute to Feature Mapping


    """

    RECOMMENDER_NAME = "BPRMF_AFM"

    def __init__(self, URM_train, ICM, **key_args):
        super(BPRMF_AFM, self).__init__(URM_train, algorithm_name="BPRMF_AFM", **key_args)
        self.ICM = check_matrix(ICM, "csr")
        self.n_features = self.ICM.shape[1]

    def fit(self, epochs=300, batch_size=128, num_factors=10, positive_threshold_BPR=None,
            learning_rate=0.01, sgd_mode='sgd', user_reg=0.0, feature_reg=0.0,
            init_mean=0.0, init_std_dev=0.1,
            stop_on_validation=False, lower_validations_allowed=None,
            validation_metric="MAP", evaluator_object=None, validation_every_n=None):
        self.model_parameters = locals()
        self.num_factors = num_factors
        self.sgd_mode = sgd_mode
        self.batch_size = batch_size
        self.positive_threshold_BPR = positive_threshold_BPR
        self.learning_rate = learning_rate

        URM_train_positive = self.URM_train.copy()
        ICM = self.ICM.copy()

        if self.positive_threshold_BPR is not None:
            URM_train_positive.data = URM_train_positive.data >= self.positive_threshold_BPR
            URM_train_positive.eliminate_zeros()

            assert URM_train_positive.nnz > 0, \
                "MatrixFactorization_Cython: URM_train_positive is empty, positive threshold is too high"

        items_to_keep = np.arange(self.n_items)[np.ediff1d(URM_train_positive.tocsc().indptr) > 0]
        self.features_to_keep = np.arange(self.n_features)[np.ediff1d(ICM[items_to_keep, :].tocsc().indptr) > 0]

        from .Cython.BPRMF_AFM_Cython_epoch import BPR_AFM_Cython_Epoch

        self.cythonEpoch = BPR_AFM_Cython_Epoch(URM_train_positive.tocsr(), ICM[:, self.features_to_keep],
                                                n_factors=self.num_factors,
                                                learning_rate=learning_rate,
                                                batch_size=1,
                                                sgd_mode=sgd_mode,
                                                init_mean=init_mean,
                                                init_std_dev=init_std_dev,
                                                user_reg=user_reg,
                                                feature_reg=feature_reg)

        self._prepare_model_for_validation()
        self._update_best_model()

        self._train_with_early_stopping(epochs,
                                        validation_every_n=validation_every_n,
                                        stop_on_validation=stop_on_validation,
                                        validation_metric=validation_metric,
                                        lower_validations_allowed=lower_validations_allowed,
                                        evaluator_object=evaluator_object,
                                        algorithm_name=self.RECOMMENDER_NAME
                                        )

        self.USER_factors = self.USER_factors_best
        self.ITEM_factors = self.ITEM_factors_best

        sys.stdout.flush()

    def _prepare_model_for_validation(self):
        self.USER_factors = self.cythonEpoch.get_USER_factors()
        self.ITEM_factors = self.ICM[:, self.features_to_keep].dot(self.cythonEpoch.get_ITEM_factors())

    def _update_best_model(self):
        self.USER_factors_best = self.USER_factors.copy()
        self.ITEM_factors_best = self.ICM[:, self.features_to_keep].dot(self.ITEM_factors.copy())
