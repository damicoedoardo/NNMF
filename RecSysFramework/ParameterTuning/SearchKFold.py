'''
Created on Sat Sep 28 2019

@author XXX
'''

from RecSysFramework.ParameterTuning.SearchOneFold import SearchOneFold, get_result_string_evaluate_on_validation
from RecSysFramework.Utils.Common import avgDicts
import time
import copy


class SearchKFold(SearchOneFold):

    ALGORITHM_NAME = "SearchKFold"

    def __init__(self, split, *pos_args, **key_args):
        """same as SearchOneFold, but it trains k models and then the score is given by the avg on
        the kfolds 

        Arguments:
            split {list} -- list of tuples returned by an object kfold splitter. [(trainfold1, testfold1), (trainfold2, testfold2), ...]
        """

        self.split = split
        self.n_folds = len(split)

        super(SearchKFold, self).__init__(*pos_args, **key_args)

    def _evaluate_on_validation(self, current_fit_parameters):

        start_time = time.time()

        result_dicts = []

        print("{}: Testing config:".format(
            self.ALGORITHM_NAME), current_fit_parameters)

        for fold in range(self.n_folds):
            print("fold {}".format(fold))
            # set up the data to be used as the train and test of the actual fold
            self.recommender_input_args.CONSTRUCTOR_KEYWORD_ARGS['URM_train'] = self.split[fold][0].get_URM(
            )
            self.evaluator_validation.global_setup(
                self.split[fold][1].get_URM())

            # Construct a new recommender instance
            recommender_instance = self.recommender_class(*self.recommender_input_args.CONSTRUCTOR_POSITIONAL_ARGS,
                                                          **self.recommender_input_args.CONSTRUCTOR_KEYWORD_ARGS)

            if fold == 0:
                recommender_instance.fit(*self.recommender_input_args.FIT_POSITIONAL_ARGS,
                                        **self.recommender_input_args.FIT_KEYWORD_ARGS,
                                        **current_fit_parameters)
            else:
                recommender_instance.fit(*self.recommender_input_args.FIT_POSITIONAL_ARGS,
                                        **FIT_KEYWORD_ARGS,
                                        **current_fit_parameters)

            # fix the epochs for the next folds to the best num of epochs found right now
            FIT_KEYWORD_ARGS = copy.deepcopy(
                self.recommender_input_args.FIT_KEYWORD_ARGS)
            if 'epochs' in self.recommender_input_args.FIT_KEYWORD_ARGS:
                FIT_KEYWORD_ARGS['epochs'] = recommender_instance.epochs_best
            if 'validation_every_n' in self.recommender_input_args.FIT_KEYWORD_ARGS:
                FIT_KEYWORD_ARGS['validation_every_n'] = recommender_instance.epochs_best

            train_time = time.time() - start_time
            start_time = time.time()

            # Evaluate recommender and get results for the first cutoff
            metric_handler = self.evaluator_validation.evaluateRecommender(
                recommender_instance)
            result_dict = metric_handler.get_results_dictionary()
            result_dict = result_dict[list(result_dict.keys())[0]]
            result_dicts.append(result_dict)

            evaluation_time = time.time() - start_time

            result_string = get_result_string_evaluate_on_validation(
                result_dict, n_decimals=7)

        # avarage the folds
        result_dict = avgDicts(result_dicts)

        return result_dict, result_string, recommender_instance, train_time, evaluation_time
