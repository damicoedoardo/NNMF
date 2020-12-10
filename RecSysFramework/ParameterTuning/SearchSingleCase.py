#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/12/18

@author: XXX, XXX
"""

import os
from RecSysFramework.Recommender.DataIO import DataIO
from RecSysFramework.ParameterTuning.SearchAbstractClass import writeLog
from RecSysFramework.ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt



class SearchSingleCase(SearchBayesianSkopt):

    ALGORITHM_NAME = "SearchSingleCase"

    def __init__(self, recommender_class, evaluator_validation=None, evaluator_test=None):

        super(SearchSingleCase, self).__init__(recommender_class,
                                               evaluator_validation=evaluator_validation,
                                               evaluator_test=evaluator_test)


    def search(self, recommender_input_args,
               fit_hyperparameters_values=None,
               metric_to_optimize="MAP",
               output_folder_path=None,
               output_file_name_root=None,
               save_metadata=True,
               recommender_input_args_last_test=None,
               ):

        assert fit_hyperparameters_values is not None, "{}: fit_hyperparameters_values must contain a dictionary".format(self.ALGORITHM_NAME)

        self.recommender_input_args = recommender_input_args
        self.recommender_input_args_last_test = recommender_input_args_last_test
        self.metric_to_optimize = metric_to_optimize
        self.output_folder_path = output_folder_path
        self.output_file_name_root = output_file_name_root
        self.resume_from_saved = False

        # If directory does not exist, create
        if not os.path.exists(self.output_folder_path):
            os.makedirs(self.output_folder_path)

        self.log_file = open(self.output_folder_path + self.output_file_name_root + "_{}.txt".format(self.ALGORITHM_NAME), "a")

        self.save_metadata = save_metadata
        self.n_calls = 1
        self.model_counter = 0
        self.save_model = "best"

        self.hyperparams_names = {}
        self.hyperparams_single_value = {}

        # In case of earlystopping the best_solution_hyperparameters will contain also the number of epochs
        self.best_solution_parameters = fit_hyperparameters_values.copy()

        self._init_metadata_dict()

        if self.save_metadata:
            self.dataIO = DataIO(folder_path = self.output_folder_path)

        self._objective_function(fit_hyperparameters_values)

        writeLog("{}: Search complete. Best config is {}: {}\n"
                 .format(self.ALGORITHM_NAME,
                         self.metadata_dict["hyperparameters_best_index"],
                         self.metadata_dict["hyperparameters_best"]),
                 self.log_file)

        if self.recommender_input_args_last_test is not None:
            self._evaluate_on_test_with_data_last()

