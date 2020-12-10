#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/2018

@author: XXX
"""

import traceback

from RecSysFramework.Recommender.KNN import ItemKNNCBF
from RecSysFramework.Recommender.NonPersonalized import TopPop

from RecSysFramework.Evaluation import EvaluatorHoldout

from RecSysFramework.DataManager.Reader import Movielens1MReader



def run_dataset(dataset_class):


    try:
        dataset_object = dataset_class()

        from RecSysFramework.DataManager.DataSplitter_leave_k_out import DataSplitter_leave_k_out
        from RecSysFramework.DataManager import DataReaderPostprocessing_K_Cores
        from RecSysFramework.DataManager import DataReaderPostprocessing_Implicit_URM

        dataset_object = DataReaderPostprocessing_K_Cores(dataset_object, k_cores_value=5)
        dataset_object = DataReaderPostprocessing_Implicit_URM(dataset_object)
        #dataset_object.load_data()



        #dataSplitter = DataSplitter_Warm_k_fold(dataset_object)
        dataSplitter = DataSplitter_leave_k_out(dataset_object, k_value=5)

        dataSplitter.load_data()

        URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()

        #
        # dataSplitter = DataSplitter_ColdItems_k_fold(dataset_object)
        #
        # dataSplitter.load_data()
        #
        # URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()
        #

        return

        evaluator = EvaluatorHoldout(URM_test, [5], exclude_seen=True)


        recommender = TopPop(URM_train)
        recommender.fit()
        _, results_run_string = evaluator.evaluateRecommender(recommender)

        log_file.write("On dataset {} - TopPop\n".format(dataset_class))
        log_file.write(results_run_string)
        log_file.flush()


        for ICM_name in dataSplitter.get_loaded_ICM_names():

            ICM_object = dataSplitter.get_ICM_from_name(ICM_name)

            recommender = ItemKNNCBFRecommender(ICM_object, URM_train)
            recommender.fit()
            _, results_run_string = evaluator.evaluateRecommender(recommender)

            log_file.write("On dataset {} - ICM {}\n".format(dataset_class, ICM_name))
            log_file.write(results_run_string)
            log_file.flush()


        log_file.write("On dataset {} PASS\n\n\n".format(dataset_class))
        log_file.flush()


    except Exception as e:

        print("On dataset {} Exception {}".format(dataset_class, str(e)))
        log_file.write("On dataset {} Exception {}\n\n\n".format(dataset_class, str(e)))
        log_file.flush()

        traceback.print_exc()


if __name__ == '__main__':

    log_file_name = "./run_test_datasets.txt"


    dataset_list = [
        Movielens1MReader,
    ]

    log_file = open(log_file_name, "a")

    for dataset_class in dataset_list:
        run_dataset(dataset_class)

