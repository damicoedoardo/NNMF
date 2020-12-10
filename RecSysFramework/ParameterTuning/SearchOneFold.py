#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 16/09/19

@author: XXX, XXX
"""

from RecSysFramework.Recommender.DataIO import DataIO

import time
import os
import traceback
import numpy as np
import sys

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
import RecSysFramework.Utils.telegram_bot as Arya

from RecSysFramework.Utils.EarlyStopping import EarlyStoppingModel

from .SearchAbstractClass import SearchAbstractClass, writeLog, get_result_string_evaluate_on_validation
from RecSysFramework.Utils.Log import dict_to_string


def _compute_avg_time_non_none_values(data_list):

    non_none_values = sum([value is not None for value in data_list])
    total_value = sum(
        [value if value is not None else 0.0 for value in data_list])

    return total_value, total_value/non_none_values


class SearchOneFold(SearchAbstractClass):

    ALGORITHM_NAME = "SearchOneFold"

    _SAVE_MODEL_VALUES = ["all", "best", "last", "no"]

    # Value to be assigned to invalid configuration or if an Exception is raised
    INVALID_CONFIG_VALUE = np.finfo(np.float16).max

    def __init__(self, recommender_class, search_iteration_strategy, evaluator_validation=None, evaluator_test=None):

        super(SearchOneFold, self).__init__(recommender_class,
                                            search_iteration_strategy,
                                            evaluator_validation=evaluator_validation,
                                            evaluator_test=evaluator_test)

    def _init_metadata_dict(self):

        self.metadata_dict = {"algorithm_name_search": self.ALGORITHM_NAME,
                              "algorithm_name_recommender": self.recommender_class.RECOMMENDER_NAME,
                              "exception_list": [None]*self.n_calls,

                              "hyperparameters_list": [None]*self.n_calls,
                              "hyperparameters_best": None,
                              "hyperparameters_best_index": None,

                              "result_on_validation_list": [None]*self.n_calls,
                              "result_on_validation_best": None,
                              "result_on_test_list": [None]*self.n_calls,
                              "result_on_test_best": None,

                              "time_on_train_list": [None]*self.n_calls,
                              "time_on_train_total": 0.0,
                              "time_on_train_avg": 0.0,

                              "time_on_validation_list": [None]*self.n_calls,
                              "time_on_validation_total": 0.0,
                              "time_on_validation_avg": 0.0,

                              "time_on_test_list": [None]*self.n_calls,
                              "time_on_test_total": 0.0,
                              "time_on_test_avg": 0.0,

                              "result_on_last": None,
                              "time_on_last_train": None,
                              "time_on_last_test": None,
                              }

    def _resume_from_saved(self):

        try:

            self.metadata_dict = self.dataIO.load_data(
                file_name=self.output_file_name_root + "_metadata")

        except FileNotFoundError:
            print("{}: Resuming '{}' Failed, no such file exists.".format(
                self.ALGORITHM_NAME, self.output_file_name_root))
            return None, None

        # Get hyperparameter list and corresponding result
        # Make sure that the hyperparameters only contain those given as input and not others like the number of epochs
        # selected by earlystopping

        hyperparameters_list_saved = self.metadata_dict['hyperparameters_list']
        result_on_validation_list_saved = self.metadata_dict['result_on_validation_list']

        hyperparameters_list_input = []
        result_on_validation_list_input = []

        while self.model_counter < len(hyperparameters_list_saved) and \
                hyperparameters_list_saved[self.model_counter] is not None:

            hyperparameters_config_saved = hyperparameters_list_saved[self.model_counter]

            hyperparameters_config_input = []

            # Add only those having a search space, in the correct ordering
            for index in range(len(self.hyperparams_names)):
                key = self.hyperparams_names[index]
                value_saved = hyperparameters_config_saved[key]

                # Check if single value categorical. It is aimed at intercepting
                # Hyperparameters that are chosen via early stopping and set them as the
                # maximum value as per hyperparameter search space. If not, the gp_minimize will return an error
                # as some values will be outside (lower) than the search space

                if isinstance(self.hyperparams_values[index], Categorical) and self.hyperparams_values[index].transformed_size == 1:
                    value_input = self.hyperparams_values[index].bounds[0]
                else:
                    value_input = value_saved

                hyperparameters_config_input.append(value_input)

            # hyperparameters_config_input = [hyperparameters_config_saved[key] for key in self.hyperparams_names]

            hyperparameters_list_input.append(hyperparameters_config_input)

            result_on_validation_list_input.append(
                - result_on_validation_list_saved[self.model_counter][self.metric_to_optimize])

            self.model_counter += 1

        print("{}: Resuming '{}'... Loaded {} configurations.".format(
            self.ALGORITHM_NAME, self.output_file_name_root, self.model_counter))

        return hyperparameters_list_input, result_on_validation_list_input

    def search(self, recommender_input_args,
               parameter_search_space,
               metric_to_optimize="MAP",
               n_cases=20,
               output_folder_path='EvaluationResults',
               output_file_name_root=None,
               save_model="best",
               save_metadata=True,
               resume_from_saved=False,
               recommender_input_args_last_test=None,
               telegram_bot=True,
               dataset_name='unknown',
               ):
        """

        :param recommender_input_args:
        :param parameter_search_space:
        :param metric_to_optimize:
        :param n_cases:
        :param n_random_starts:
        :param output_folder_path:
        :param output_file_name_root:
        :param save_model:          "no"    don't save anything
                                    "all"   save every model
                                    "best"  save the best model trained on train data alone and on last, if present
                                    "last"  save only last, if present
        :param save_metadata:
        :param recommender_input_args_last_test:
        :return:
        """

        if save_model not in self._SAVE_MODEL_VALUES:
            raise ValueError("{}: parameter save_model must be in '{}', provided was '{}'.".format(
                self.ALGORITHM_NAME, self._SAVE_MODEL_VALUES, save_model))

        if save_model == "last" and recommender_input_args_last_test:
            print("{}: parameter save_model is 'last' but no recommender_input_args_last_test provided, saving best model on train data alone.".format(
                self.ALGORITHM_NAME))
            save_model = "best"

        self.save_model = save_model
        self.resume_from_saved = resume_from_saved
        self.recommender_input_args = recommender_input_args
        self.recommender_input_args_last_test = recommender_input_args_last_test
        self.parameter_search_space = parameter_search_space
        self.metric_to_optimize = metric_to_optimize
        self.output_folder_path = output_folder_path
        self.output_file_name_root = output_file_name_root
        self.n_calls = n_cases
        self.telegram_bot = telegram_bot
        self.fixed_param_dictionary = recommender_input_args.FIT_KEYWORD_ARGS
        self.dataset_name = dataset_name

        # If directory does not exist, create
        if not os.path.exists(self.output_folder_path):
            print('Creating DIR named {}\n'.format(self.output_folder_path))
            os.makedirs(self.output_folder_path)

        if self.output_file_name_root is None:
            print('WARNING: the output filename has not been specified! \n it will be {}\n'.format(
                self.recommender_class.RECOMMENDER_NAME))
            self.output_file_name_root = self.recommender_class.RECOMMENDER_NAME

        self.log_file = open(os.path.join(self.output_folder_path, self.output_file_name_root) +
                             "_{}_dataset_{}.txt".format(self.ALGORITHM_NAME, dataset_name), "a")
        print('Saving evaluation result in {}'.format(os.path.join(self.output_folder_path, self.output_file_name_root) +
                                                      "_{}_dataset_{}.txt".format(self.ALGORITHM_NAME, dataset_name)))
        self.model_counter = 0

        self.n_jobs = 1
        self.save_metadata = save_metadata

        self._init_metadata_dict()

        if self.save_metadata:
            self.dataIO = DataIO(folder_path=self.output_folder_path)

        self.hyperparams = dict()
        self.hyperparams_names = list()
        self.hyperparams_values = list()

        skopt_types = [Real, Integer, Categorical]

        for name, hyperparam in self.parameter_search_space.items():

            if any(isinstance(hyperparam, sko_type) for sko_type in skopt_types):
                self.hyperparams_names.append(name)
                self.hyperparams_values.append(hyperparam)
                self.hyperparams[name] = hyperparam

            else:
                raise ValueError("{}: Unexpected parameter type: {} - {}".format(
                    self.ALGORITHM_NAME, str(name), str(hyperparam)))

        if self.resume_from_saved:
            hyperparameters_list_input, result_on_validation_list_saved = self._resume_from_saved()
            self.x0 = hyperparameters_list_input
            self.y0 = result_on_validation_list_saved

            self.n_random_starts = max(
                0, self.n_random_starts - self.model_counter)
            self.n_calls = max(0, self.n_calls - self.model_counter)

         # print FIXED parameters
        print('Fixed Parameters of the search are:{}'.format(
            recommender_input_args.FIT_KEYWORD_ARGS))

        # call the subclass implementation
        self._search_iterations()

        writeLog("{}: Search complete. Best config is {}: {}\n"
                 .format(self.ALGORITHM_NAME,
                         self.metadata_dict["hyperparameters_best_index"],
                         self.metadata_dict["hyperparameters_best"]),
                 self.log_file)

        if self.telegram_bot:
            Arya.send_message("{}: Search complete.\n\n Best config is {}:\n\n{}\n\n {}\n"
                              .format(self.ALGORITHM_NAME,
                                      self.metadata_dict["hyperparameters_best_index"],
                                      'Fixed Parameters of the search are:{}{}'.format(recommender_input_args.FIT_POSITIONAL_ARGS,
                                                                                       recommender_input_args.FIT_KEYWORD_ARGS),
                                      self.metadata_dict["hyperparameters_best"]))

        if self.recommender_input_args_last_test is not None:
            self._evaluate_on_test_with_data_last()

    def _search_iterations(self):
        self.search_iteration_strategy.search_iteration(self)

    def _evaluate_on_validation(self, current_fit_parameters):

        start_time = time.time()

        # Construct a new recommender instance
        recommender_instance = self.recommender_class(*self.recommender_input_args.CONSTRUCTOR_POSITIONAL_ARGS,
                                                      **self.recommender_input_args.CONSTRUCTOR_KEYWORD_ARGS)

        print("{}: Testing config:".format(
            self.ALGORITHM_NAME), current_fit_parameters)

        recommender_instance.fit(*self.recommender_input_args.FIT_POSITIONAL_ARGS,
                                 **self.recommender_input_args.FIT_KEYWORD_ARGS,
                                 **current_fit_parameters)

        train_time = time.time() - start_time
        start_time = time.time()

        # Evaluate recommender and get results for the first cutoff
        metric_handler = self.evaluator_validation.evaluateRecommender(
            recommender_instance)
        result_dict = metric_handler.get_results_dictionary()
        result_dict = result_dict[list(result_dict.keys())[0]]

        evaluation_time = time.time() - start_time

        result_string = get_result_string_evaluate_on_validation(
            result_dict, n_decimals=7)

        return result_dict, result_string, recommender_instance, train_time, evaluation_time

    def _evaluate_on_test(self, recommender_instance, current_fit_parameters_dict, print_log=True):

        start_time = time.time()

        # Evaluate recommender and get results for the first cutoff
        metric_handler = self.evaluator_test.evaluateRecommender(
            recommender_instance)

        result_dict = metric_handler.get_results_dictionary()
        result_string = metric_handler.get_results_string()

        evaluation_test_time = time.time() - start_time

        if print_log:
            writeLog("{}: Best config evaluated with evaluator_test. Config: {} - results:\n{}\n"
                     .format(self.ALGORITHM_NAME, current_fit_parameters_dict, result_string),
                     self.log_file)

        return result_dict, result_string, evaluation_test_time

    def _evaluate_on_test_with_data_last(self):

        start_time = time.time()

        # Construct a new recommender instance
        recommender_instance = self.recommender_class(*self.recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS,
                                                      **self.recommender_input_args_last_test.CONSTRUCTOR_KEYWORD_ARGS)

        # Check if last was already evaluated
        if self.resume_from_saved:
            result_on_last_saved_flag = self.metadata_dict["result_on_last"] is not None and \
                self.metadata_dict["time_on_last_train"] is not None and \
                self.metadata_dict["time_on_last_test"] is not None

            try:
                recommender_instance.load_model(
                    self.output_folder_path, file_name=self.output_file_name_root + "_best_model_last")
                print("{}: Resuming '{}'... Result on last already available.".format(
                    self.ALGORITHM_NAME, self.output_file_name_root))
            except:
                result_on_last_saved_flag = False

            if result_on_last_saved_flag:
                return

        print("{}: Evaluation with constructor data for final test. Using best config:"
              .format(self.ALGORITHM_NAME), self.metadata_dict["hyperparameters_best"])

        # Use the hyperparameters that have been saved
        assert self.metadata_dict["hyperparameters_best"] is not None, "{}: Best hyperparameters not available, the search might have failed."
        fit_keyword_args = self.metadata_dict["hyperparameters_best"].copy()

        recommender_instance.fit(*self.recommender_input_args_last_test.FIT_POSITIONAL_ARGS,
                                 **fit_keyword_args)

        train_time = time.time() - start_time

        result_dict_test, result_string, evaluation_test_time = self._evaluate_on_test(
            recommender_instance, fit_keyword_args, print_log=False)

        writeLog("{}: Best config evaluated with evaluator_test with constructor data for final test. Config: {} - results:\n{}\n"
                 .format(self.ALGORITHM_NAME, self.metadata_dict["hyperparameters_best"], result_string),
                 self.log_file)

        self.metadata_dict["result_on_last"] = result_dict_test
        self.metadata_dict["time_on_last_train"] = train_time
        self.metadata_dict["time_on_last_test"] = evaluation_test_time

        if self.save_metadata:
            self.dataIO.save_data(data_dict_to_save=self.metadata_dict.copy(),
                                  file_name=self.output_file_name_root + "_metadata")

        if self.save_model in ["all", "best", "last"]:
            print("{}: Saving model in {}\n".format(self.ALGORITHM_NAME,
                                                    self.output_folder_path + self.output_file_name_root))
            recommender_instance.save_model(
                self.output_folder_path, file_name=self.output_file_name_root + "_best_model_last")

    def _objective_function_list_input(self, current_fit_parameters_list_of_values):
        current_fit_parameters_dict = dict(
            zip(self.hyperparams_names, current_fit_parameters_list_of_values))
        return self._objective_function(current_fit_parameters_dict)

    def _save_model(self, save_folder_path, model_name, recommender_instance, descr=None):
        recommender_instance.save_model(save_folder_path, file_name=model_name)
        if descr is not None:
            with open('{}_descr.txt'.format(save_folder_path), 'w') as f:
                print('params: ', descr, file=f)

    def _objective_function(self, current_fit_parameters_dict):

        try:

            result_dict, result_string, recommender_instance, train_time, evaluation_time = self._evaluate_on_validation(
                current_fit_parameters_dict)

            current_result = - result_dict[self.metric_to_optimize]

            # If the recommender uses Earlystopping, get the selected number of epochs
            if isinstance(recommender_instance, EarlyStoppingModel):

                n_epochs_early_stopping_dict = recommender_instance.get_early_stopping_final_epochs_dict()
                current_fit_parameters_dict = current_fit_parameters_dict.copy()

                for epoch_label in n_epochs_early_stopping_dict.keys():

                    epoch_value = n_epochs_early_stopping_dict[epoch_label]
                    current_fit_parameters_dict[epoch_label] = epoch_value

            # set the paths for the model data
            save_folder_path = os.path.join(
                self.output_folder_path, 'Saved_Models_{}_{}'.format(self.output_file_name_root, self.dataset_name), dict_to_string(result_dict))
            model_name = None

            # Always save best model separately
            if self.save_model in ["all"]:
                self._save_model(save_folder_path, model_name, recommender_instance, '{} {}'.format(dict_to_string(self.fixed_param_dictionary, style='constructor'),
                                                                                                    dict_to_string(current_fit_parameters_dict, style='constructor')))

            if self.metadata_dict["result_on_validation_best"] is None:
                new_best_config_found = True
            else:
                best_solution_val = self.metadata_dict["result_on_validation_best"][self.metric_to_optimize]
                new_best_config_found = best_solution_val < result_dict[self.metric_to_optimize]

            if new_best_config_found:

                writeLog("{}: New best config found. Config {}: {} - results: {}\n".format(self.ALGORITHM_NAME,
                                                                                           self.model_counter,
                                                                                           current_fit_parameters_dict,
                                                                                           result_string), self.log_file)

                if self.telegram_bot:
                    message = 'Recommender {}, validating {} with sample strategy {} on {}. \n\nFixed params: {}\n\nSpace params: {}\n\nResult: {}\n\nElapsed time: {}'.format(
                        self.metadata_dict['algorithm_name_recommender'],
                        self.metadata_dict['algorithm_name_search'],
                        self.search_iteration_strategy.ITERATION_STRATEGY,
                        self.dataset_name,
                        dict_to_string(
                            self.fixed_param_dictionary, style='constructor'),
                        dict_to_string(
                            current_fit_parameters_dict, style='constructor'),
                        result_string,
                        '{} min'.format(
                            round(train_time/60, 2)),
                    )
                    Arya.send_message(message)

                if self.save_model in ["all", "best"]:
                    self._save_model(save_folder_path,
                                     model_name, recommender_instance, '{} {}'.format(dict_to_string(self.fixed_param_dictionary, style='constructor'),
                                                                                      dict_to_string(current_fit_parameters_dict, style='constructor')))
                if self.evaluator_test is not None:
                    result_dict_test, _, evaluation_test_time = self._evaluate_on_test(
                        recommender_instance, current_fit_parameters_dict, print_log=True)

            else:
                writeLog("{}: Config {} is suboptimal. Config: {} - results: {}\n".format(self.ALGORITHM_NAME,
                                                                                          self.model_counter,
                                                                                          current_fit_parameters_dict,
                                                                                          result_string), self.log_file)

            if current_result >= self.INVALID_CONFIG_VALUE:
                writeLog("{}: WARNING! Config {} returned a value equal or worse than the default value to be assigned to invalid configurations."
                         " If no better valid configuration is found, this parameter search may produce an invalid result.\n", self.log_file)

            self.metadata_dict["hyperparameters_list"][self.model_counter] = current_fit_parameters_dict.copy(
            )
            self.metadata_dict["result_on_validation_list"][self.model_counter] = result_dict.copy(
            )

            self.metadata_dict["time_on_train_list"][self.model_counter] = train_time
            self.metadata_dict["time_on_validation_list"][self.model_counter] = evaluation_time

            self.metadata_dict["time_on_train_total"], self.metadata_dict["time_on_train_avg"] = _compute_avg_time_non_none_values(
                self.metadata_dict["time_on_train_list"])
            self.metadata_dict["time_on_validation_total"], self.metadata_dict["time_on_validation_avg"] = _compute_avg_time_non_none_values(
                self.metadata_dict["time_on_validation_list"])

            if new_best_config_found:
                self.metadata_dict["hyperparameters_best"] = current_fit_parameters_dict.copy(
                )
                self.metadata_dict["hyperparameters_best_index"] = self.model_counter
                self.metadata_dict["result_on_validation_best"] = result_dict.copy(
                )

                if self.evaluator_test is not None:
                    self.metadata_dict["result_on_test_best"] = result_dict_test.copy(
                    )
                    self.metadata_dict["result_on_test_list"][self.model_counter] = result_dict_test.copy(
                    )
                    self.metadata_dict["time_on_test_list"][self.model_counter] = evaluation_test_time

                    self.metadata_dict["time_on_test_total"], self.metadata_dict["time_on_test_avg"] = _compute_avg_time_non_none_values(
                        self.metadata_dict["time_on_test_list"])

        except:
            # Catch any error: Exception, Tensorflow errors etc...

            traceback_string = traceback.format_exc()

            writeLog("{}: Config {} Exception. Config: {} - Exception: {}\n".format(self.ALGORITHM_NAME,
                                                                                    self.model_counter,
                                                                                    current_fit_parameters_dict,
                                                                                    traceback_string), self.log_file)

            self.metadata_dict["exception_list"][self.model_counter] = traceback_string

            # Assign to this configuration the worst possible score
            # Being a minimization problem, set it to the max value of a float
            current_result = + self.INVALID_CONFIG_VALUE

            traceback.print_exc()

            sys.exit()

        if self.save_metadata:
            self.dataIO.save_data(data_dict_to_save=self.metadata_dict.copy(),
                                  file_name=self.output_file_name_root + "_metadata")

        self.model_counter += 1

        return current_result
