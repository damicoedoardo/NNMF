#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: XXX
"""

import numpy as np
import similaripy as sim

from RecSysFramework.Recommender import ItemSimilarityMatrixRecommender
from RecSysFramework.Utils import check_matrix

from RecSysFramework.Utils.FeatureWeighting import okapi_BM_25, TF_IDF
from tkinter.filedialog import askopenfilename


class ItemKNNCF(ItemSimilarityMatrixRecommender):

    """ ItemKNN recommender"""

    RECOMMENDER_NAME = "ItemKNNCF"

    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]


    def __init__(self, URM_train):
        super(ItemKNNCF, self).__init__(URM_train)


    def fit(self, topK=50, shrink=100, similarity='cosine', feature_weighting="none", **similarity_args):

        # Similaripy returns also self similarity, which will be set to 0 afterwards
        topK += 1
        self.topK = topK
        self.shrink = shrink

        if feature_weighting not in self.FEATURE_WEIGHTING_VALUES:
            raise ValueError("Value for 'feature_weighting' not recognized. Acceptable values are {}, provided was '{}'"
                             .format(self.FEATURE_WEIGHTING_VALUES, feature_weighting))

        if feature_weighting == "BM25":
            self.URM_train = self.URM_train.astype(np.float32)
            self.URM_train = okapi_BM_25(self.URM_train.T).T
            self.URM_train = check_matrix(self.URM_train, 'csr')

        elif feature_weighting == "TF-IDF":
            self.URM_train = self.URM_train.astype(np.float32)
            self.URM_train = TF_IDF(self.URM_train.T).T
            self.URM_train = check_matrix(self.URM_train, 'csr')

        if similarity == "cosine":
            self.W_sparse = sim.cosine(self.URM_train.T, k=topK, shrink=shrink, **similarity_args)
        elif similarity == "jaccard":
            self.W_sparse = sim.jaccard(self.URM_train.T, k=topK, shrink=shrink, **similarity_args)
        elif similarity == "dice":
            self.W_sparse = sim.dice(self.URM_train.T, k=topK, shrink=shrink, **similarity_args)
        elif similarity == "tversky":
            self.W_sparse = sim.tversky(self.URM_train.T, k=topK, shrink=shrink, **similarity_args)
        elif similarity == "splus":
            self.W_sparse = sim.s_plus(self.URM_train.T, k=topK, shrink=shrink, **similarity_args)
        else:
            raise ValueError("Unknown value '{}' for similarity".format(similarity))

        self.W_sparse.setdiag(0)
        self.W_sparse = self.W_sparse.transpose().tocsr()

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
        super(ItemKNNCF, self).load_model(folder_path=folder_path, file_name=file_name)