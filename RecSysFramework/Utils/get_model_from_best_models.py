from RecSysFramework.Utils import menu
from RecSysFramework.Utils.get_holdout import retrieve_train_validation_test_holdhout_dataset
from RecSysFramework.Recommender.MatrixFactorization.MatrixFactorization_Cython import BPRMF, BPR_NNMF, FunkSVD, FUNK_NNMF, PROB_NNMF
from RecSysFramework.Recommender.SLIM.BPR.SLIM_BPR_Cython import SLIM_BPR_Cython
from RecSysFramework.Recommender.MatrixFactorization.PureSVD import PureSVD
from RecSysFramework.Recommender.KNN.ItemKNNCFRecommender import ItemKNNCF
from RecSysFramework.Recommender.KNN.UserKNNCFRecommender import UserKNNCF
import os
import numpy as np


def get_model(train, dataset_name, model_name=None, caption='Select model'):
    if model_name == None:
        model_name = menu.single_choice(caption,
                                        ['BPRMF', 'BPR_NNMF', 'FunkSVD', 'FUNK_NNMF', 'ItemKNNCF', 'UserKNNCF', 'nnprobmf', 'probmf', 'PureSVD', 'SLIM_BPR_Cython'])
    if model_name not in ['nnprobmf', 'probmf']:
        m1 = eval(model_name)(train.get_URM())
        m1.load_model(
            folder_path='BestModels/{}/{}/'.format(dataset_name, model_name))
    else:
        m1 = PROB_NNMF(train.get_URM())

        filenames = np.array(os.listdir('BestModels/{}/{}'.format(dataset_name, model_name)))
        # mantain only the filename with .zip in the file
        mask = np.array(['.zip' in fn for fn in filenames])
        filename = filenames[mask][0].split('.zip')[0]

        m1.load_model(
            folder_path='BestModels/{}/{}/'.format(dataset_name, model_name), file_name=filename)

        if model_name == 'probmf':
            model_name = 'ProbMF'
        elif model_name == 'nnprobmf':
            model_name = 'Prob_NNMF'

    return m1, model_name.replace('SLIM_BPR_Cython', 'SLIM_BPR').replace('_', ' ').replace('FUNK', 'Funk')


if __name__ == '__main__':
    train, test, validation, name = retrieve_train_validation_test_holdhout_dataset()
    get_model(train, name)
