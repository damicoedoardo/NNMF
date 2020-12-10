# Reproducibility1: Dataset loader splitting and training of the model

import RecSysFramework.Utils.get_holdout as gh
from RecSysFramework.Recommender.MatrixFactorization.MatrixFactorization_Cython import (
    BPRMF,
    BPR_NNMF,
    FunkSVD,
    FUNK_NNMF,
    PROB_NNMF,
)
from RecSysFramework.Recommender.SLIM.BPR.SLIM_BPR_Cython import SLIM_BPR_Cython
from RecSysFramework.Recommender.MatrixFactorization.PureSVD import PureSVD
from RecSysFramework.Recommender.KNN.ItemKNNCFRecommender import ItemKNNCF
from RecSysFramework.Recommender.KNN.UserKNNCFRecommender import UserKNNCF
from RecSysFramework.Evaluation.Evaluator import EvaluatorHoldout, EvaluatorMetrics

# define the datasets used in the experiments

from param_read import get_best_params


def split_train_save(ex_dataset, ex_model_class):
    """
    For each dataset:
        1) download the dataset
        2) split the dataset in Train, validation and test URM
        For each model:
            a) retrieve the best parameters from BestModels/...
            b) train the model on the train URM
            c) evaluate it on the test URM
            d) save the model in BestModels
    """
    for d in ex_dataset:
        # get the URMs
        train, _, test, d_name = gh.retrieve_train_validation_test_holdhout_dataset(d)
        print("Dataset: {}".format(d))
        train_urm = train.get_URM()
        test_urm = test.get_URM()

        # instantiate the evaluator
        evaluator = EvaluatorHoldout(cutoff_list=[5], metrics_list=[EvaluatorMetrics.MAP])

        for m_c in ex_model_class:
            # implementation of prob_nnmf and probmf same class
            if m_c in ['PROB_NNMF', 'PROBMF']:
                m_name = 'nnprobmf' if m_c == 'PROB_NNMF' else 'probmf'
                m_c = PROB_NNMF
            else:
                m_name = m_c.__name__

            # instantiate the model
            m = m_c(train_urm)
            # get the best parameters
            params = get_best_params(d_name, m_name)
            # fit the model
            m.fit(**params)
            # evaluate the recommender
            res = evaluator.evaluateRecommender(m, test_urm).get_results_string()
            print("Model: {}, Result: {}\n".format(m_name, res))
            save_folder = "BestModels/{}/{}/".format(d_name, m_name)
            if m_name == "SLIM_BPR_Cython":
                m_name = m_c.RECOMMENDER_NAME
            m.save_model(save_folder, m_name, save_description=False)


if __name__ == '__main__':
    ex_dataset = [
        "Movielens1MReader",
        "LastFMHetrec2011Reader",
        "BookCrossingReader",
        "PinterestReader",
        "CiteULike_aReader",
    ]

    # define the model classes used in the experiments
    ex_model_class = [
        ItemKNNCF,
        UserKNNCF,
        PureSVD,
        SLIM_BPR_Cython,
        BPRMF,
        BPR_NNMF,
        FunkSVD,
        FUNK_NNMF,
        'PROBMF',
        'PROB_NNMF',
    ]

    split_train_save(ex_dataset, ex_model_class)
