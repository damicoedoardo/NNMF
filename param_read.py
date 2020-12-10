import numpy as np
import os
import RecSysFramework.Utils.menu as menu
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
from functools import partial


def _read_params_file(dataset_name, model_name):
    filenames = np.array(
        os.listdir("BestModels/{}/{}".format(dataset_name, model_name))
    )
    # mantain only the filename with .txt in the file
    mask = np.array([".txt" in fn for fn in filenames])
    filename = filenames[mask][0]  # .split('.txt')[0]
    params_file = open(
        "BestModels/{}/{}/{}".format(dataset_name, model_name, filename), "r"
    )
    return params_file.read()


def get_best_params(dataset_name, model_name):
    """
    @return: dictionary with the best parameters
    """

    def get_model_param_list(m):
        """
        Given a model return a list containing the model parameters
        @param m:

        @return:
        """
        if m in ["ItemKNNCF", "UserKNNCF"]:
            return ["topK", "shrink"]
        elif m == "PureSVD":
            return ["num_factors"]
        elif m == "SLIM_BPR_Cython":
            return [
                "sgd_mode",
                "epochs",
                "symmetric",
                "train_with_sparse_weights",
                "batch_size",
                "lambda_i",
                "lambda_j",
                "learning_rate",
                "topK",
                "gamma",
                "beta_1",
                "beta_2",
            ]
        # todo: has been tune with nnmf check if the results still hold
        elif m == "BPRMF":
            return [
                "sgd_mode",
                "use_bias",
                "batch_size",
                "num_factors",
                "learning_rate",
                "user_reg",
                "positive_reg",
                "negative_reg",
                "epochs",
                "latent_factors_initialization",
            ]
        elif m == "BPR_NNMF":
            return [
                "sgd_mode",
                "use_bias",
                "batch_size",
                "num_factors",
                "learning_rate",
                "item_reg",
                "user_reg",
                "positive_reg",
                "negative_reg",
                "epochs",
                "user_eye",
                "item_eye",
                "item_k",
                "user_k",
                "item_shrink",
                "user_shrink",
                "latent_factors_initialization",
            ]
        elif m == "FunkSVD":
            return [
                "sgd_mode",
                "use_bias",
                "batch_size",
                "num_factors",
                "learning_rate",
                "item_reg",
                "user_reg",
                "negative_interactions_quota",
                "epochs",
                "latent_factors_initialization",
            ]
        elif m == "FUNK_NNMF":
            return [
                "sgd_mode",
                "use_bias",
                "batch_size",
                "num_factors",
                "learning_rate",
                "item_reg",
                "user_reg",
                "negative_interactions_quota",
                "epochs",
                "item_k",
                "user_k",
                "item_shrink",
                "user_shrink",
                "latent_factors_initialization",
            ]
        # TODO: understand how to distinguish between nnmf and standard prob
        elif m in ["nnprobmf", "probmf"]:
            return [
                "sgd_mode",
                "use_bias",
                "batch_size",
                "num_factors",
                "learning_rate",
                "item_reg",
                "user_reg",
                "negative_interactions_quota",
                "epochs",
                "item_k",
                "user_k",
                "item_shrink",
                "user_shrink",
                "latent_factors_initialization",
            ]

    best_params = {}
    file = _read_params_file(dataset_name, model_name)
    param_list = get_model_param_list(model_name)
    for p in param_list:
        best_params[p] = get_param(file, p)
    if model_name in [
        'BPRMF',
        'BPR_NNMF',
        'FunkSVD',
        'FUNK_NNMF',
        "nnprobmf",
        "probmf",
    ]:
        best_params["verbose"] = True
    return best_params


def get_param(params_string, param_name):
    if param_name == "latent_factors_initialization":
        return eval(params_string.split(param_name + "=")[1].split("}")[0] + "}")
    elif param_name == "epochs":
        return int(params_string.split(param_name + "=")[-1].split(",")[0])
    elif param_name == "sgd_mode":
        return params_string.split(param_name + "=")[-1].split(",")[0]
    else:
        return eval(params_string.split(param_name + "=")[1].split(",")[0])


