import pandas as pd
import numpy as np
import gc
from RecSysFramework.Utils.check_folder import check_folder
from RecSysFramework.Utils.DatasetCreator import create_dataset
from RecSysFramework.DataManager.Reader.Movielens1MReader import Movielens1MReader
from RecSysFramework.DataManager.Reader.LastFMHetrec2011Reader import LastFMHetrec2011Reader
from RecSysFramework.DataManager.Reader.BookCrossingReader import BookCrossingReader
from RecSysFramework.DataManager.Reader.CiteULikeReader import CiteULike_aReader
from RecSysFramework.DataManager.Splitter import Holdout
from RecSysFramework.DataManager.Splitter import KFold
from RecSysFramework.DataManager.DatasetPostprocessing.ImplicitURM import ImplicitURM
# from RecSysFramework.DataManager.DatasetPostprocessing.LongQueueAnalysis import LongQueueAnalysis
from RecSysFramework.DataManager.DatasetPostprocessing.KCore import KCore
from RecSysFramework.DataManager.Splitter.KFold import WarmItemsKFold
import RecSysFramework.Utils.menu as menu
from RecSysFramework.DataManager.Reader.PinterestReader import PinterestReader
import os
import random
from RecSysFramework.Recommender.MatrixFactorization.MatrixFactorization_Cython import BPR_NNMF, FUNK_NNMF, PROB_NNMF,\
    BPRMF, FunkSVD
from functools import partial
from RecSysFramework.Recommender import DataIO
import scipy.sparse as sps


def create_and_save_k_split(dataset_name, k=10):
    base_save_folder_path = 'SignificanceTest'

    # add the dataset name to the main folder path
    base_save_folder_path += os.sep+dataset_name.split('Reader')[0]

    dataset_reader = eval(dataset_name)()
    dataset = dataset_reader.load_data()

    if dataset_name == 'Movielens1MReader':
        preprocessing = [ImplicitURM(3), KCore(5, 5)]
    if dataset_name == 'BookCrossingReader':
        preprocessing = [ImplicitURM(6), KCore(5, 5)]
    if dataset_name == 'CiteULike_aReader':
        preprocessing = [KCore(5, 5)]
    if dataset_name == 'LastFMHetrec2011Reader':
        preprocessing=[ImplicitURM(1), KCore(5, 5)]
    if dataset_name == 'PinterestReader':
        preprocessing=[ImplicitURM(1), KCore(5, 5)]

    for i in preprocessing:
        dataset=i.apply(dataset)

    train_perc=0.6
    val_perc=0.2
    test_perc=0.2

    # correction for citeulike
    base_save_folder_path=base_save_folder_path.replace('_a', '-a')

    random_seeds=[]
    for i in range(k):
        # set the random seed
        while True:
            random_seed=random.randint(1, 500)
            if random_seed not in random_seeds:
                random_seeds.append(random_seed)
                break
        print('random_seed:{}'.format(random_seed))

        # set the save_folder_path
        final_path=base_save_folder_path+os.sep+'splits' + \
            os.sep+'dataset_{}'.format(str(i))+os.sep
        splitter=Holdout(train_perc = train_perc, validation_perc = val_perc, test_perc = test_perc,
                           random_seed = random_seed)
        splitter.save_split(splitter.split(dataset),
                            filename_suffix = '', save_folder_path = final_path)


def _read_params_file(dataset_name, model_name):
    """
    Given a dataset and the model name
    retrieve the optimal parameters of the model and return it as dictionary

    :param dataset_name: name of the dataset
    :param model_name: name of the model
    :return: hyperparameters dictionary
    """
    filenames=np.array(os.listdir(
        'BestModels/{}/{}'.format(dataset_name, model_name)))
    # mantain only the filename with .txt in the file
    mask=np.array(['.txt' in fn for fn in filenames])
    filename=filenames[mask][0]  # .split('.txt')[0]
    params_file=open(
        'BestModels/{}/{}/{}'.format(dataset_name, model_name, filename), 'r')
    return params_file.read()


def get_param(params_string, param_name):
    if param_name == 'latent_factors_initialization':
        return eval(params_string.split(param_name + '=')[1].split('}')[0] + '}')
    elif param_name == 'epochs':
        return int(params_string.split('epochs=')[-1].split(',')[0])
    else:
        return eval(params_string.split(param_name + '=')[1].split(',')[0])


def get_params_dict_model_dataset(dataset_name, model_name):
    params_string=_read_params_file(dataset_name, model_name)
    _get_param=partial(get_param, params_string)
    if model_name == 'BPRMF' or model_name == 'BPR_NNMF':
        params_dict={
            'num_factors': _get_param('num_factors'),
            'learning_rate': _get_param('learning_rate'),
            'user_reg': _get_param('user_reg'),
            'positive_reg': _get_param('positive_reg'),
            'negative_reg': _get_param('negative_reg'),
            'latent_factors_initialization': _get_param('latent_factors_initialization'),
            'sgd_mode': 'adam',
            'validation_metric': 'MAP',
            'verbose': True,
            'use_bias': False,
            'positive_threshold_BPR': 0,
            'epochs': _get_param('epochs'),
            'batch_size': _get_param('batch_size'),
        }
        if model_name == 'BPR_NNMF':
            nn_params = {
                'item_k': _get_param('item_k'),
                'user_k': _get_param('user_k'),
                'item_reg': _get_param('item_reg'),
                'item_shrink': _get_param('item_shrink'),
                'user_shrink': _get_param('user_shrink'),
                'user_eye': False,
                'item_eye': False,
            }
            params_dict={**params_dict, **nn_params}

    elif model_name == 'FunkSVD' or model_name == 'FUNK_NNMF':
        epochs=_get_param('epochs')
        params_dict={
            'num_factors': _get_param('num_factors'),
            'learning_rate': _get_param('learning_rate'),
            'item_reg': _get_param('item_reg'),
            'user_reg': _get_param('user_reg'),
            'negative_interactions_quota': _get_param('negative_interactions_quota'),
            'latent_factors_initialization': _get_param('latent_factors_initialization'),
            'sgd_mode': 'adam',
            'validation_metric': 'MAP',
            'verbose': True,
            'use_bias': False,
            'positive_threshold_BPR': 0,
            'epochs': epochs,
            'batch_size': _get_param('batch_size'),
        }
        if model_name == 'FUNK_NNMF':
            nn_params={
                'item_k': _get_param('item_k'),
                'user_k': _get_param('user_k'),
                'item_shrink': _get_param('item_shrink'),
                'user_shrink': _get_param('user_shrink'),
                'user_eye': False,
                'item_eye': False,
            }
            params_dict={**params_dict, **nn_params}

    elif model_name == 'probmf' or model_name == 'nnprobmf':
        params_dict={
            'item_k': 1 if model_name == 'probmf' else _get_param('item_k'),
            'user_k': 1 if model_name == 'probmf' else _get_param('user_k'),
            'num_factors': _get_param('num_factors'),
            'learning_rate': _get_param('learning_rate'),
            'item_reg': _get_param('item_reg'),
            'user_reg': _get_param('user_reg'),
            'latent_factors_initialization': _get_param('latent_factors_initialization'),
            'sgd_mode': 'adam',
            'validation_metric': 'MAP',
            'verbose': True,
            'use_bias': False,
            'positive_threshold_BPR': 0,
            'epochs': _get_param('epochs'),
            'item_shrink': 0 if model_name == 'probmf' else _get_param('item_shrink'),
            'user_shrink': 0 if model_name == 'probmf' else _get_param('user_shrink'),
            'batch_size': _get_param('batch_size'),
            'user_eye': True if model_name == 'probmf' else False,
            'item_eye': True if model_name == 'probmf' else False,
        }

    return params_dict


def load_significance_split(dataset_name, dataset_number, mode):
    """

    :param dataset_number: number of the dataset to load
    :param mode: TRAIN, VAL or TEST
    :return: urm
    """
    assert mode in ['TRAIN', 'VALIDATION', 'TEST']

    # correction for the dataset CiteULike
    dataset_name=dataset_name.replace('_', '-')

    # create the split_basepath
    split_basepath='SignificanceTest/{}/splits/dataset_{}'.format(
        dataset_name, str(dataset_number))

    TRAIN_PATH='URM_all_train.npz'
    VALIDATION_PATH='URM_all_validation.npz'
    TEST_PATH='URM_all_test.npz'

    try:
        load_path=split_basepath+os.sep+os.listdir(split_basepath)[0]+os.sep
    except FileNotFoundError:
        print(f"Creating 10 splits prior to training ...")
        create_and_save_k_split(dataset_name.replace("CiteULike-a", "CiteULike_a")+"Reader", k=10) 
        load_path=split_basepath+os.sep+os.listdir(split_basepath)[0]+os.sep

    if mode == 'TRAIN':
        load_path += TRAIN_PATH
    elif mode == 'VALIDATION':
        load_path += VALIDATION_PATH
    elif mode == 'TEST':
        load_path += TEST_PATH

    return sps.load_npz(load_path)
       


def load_significance_model(dataset_name, dataset_number, model_name):
    # load Train data associated to the split number
    urm_train=load_significance_split(
        dataset_name, dataset_number, mode='TRAIN')

    # correction for citeulike dataset
    dataset_name=dataset_name.replace('_', '-')

    while True:
        try:
            if model_name == 'PROBMF':
                model=eval('PROB_NNMF')(urm_train)
                model.load_model(folder_path='SignificanceTest/{}/{}/'.format(dataset_name, model_name),
                                file_name='{}_{}.zip'.format('PROB_NNMF', dataset_number))
            else:
                model=eval(model_name)(urm_train)
                model.load_model(folder_path='SignificanceTest/{}/{}/'.format(dataset_name, model_name),
                                file_name='{}_{}.zip'.format(model_name, dataset_number))
            break
        except FileNotFoundError:
            print("model missing, fitting ...")
            _train_and_save_models(dataset_name, model_name, 10)
    
    return model

def _train_and_save_models(dataset_name, model_name, k=10):
    # TODO: change folder name of best models related to the prob_nnmf
    # correction to the name of the probnnmf
    model_name_fixed=model_name
    if model_name_fixed == 'PROB_NNMF':
        model_name_fixed='nnprobmf'
    elif model_name_fixed == 'PROBMF':
        model_name_fixed='probmf'

    # load the best hyperparameters of the model
    params_dict=get_params_dict_model_dataset(dataset_name, model_name_fixed)

    save_model_base_path='SignificanceTest/{}/{}/'.format(
        dataset_name, model_name)
    check_folder(save_model_base_path)

    # fix since probmf and nnprobmf are built using same class
    if model_name == 'PROBMF':
        model_name='PROB_NNMF'

    for split_num in range(k):
        print('Train model {}\n can take hours...'.format(split_num))
        urm_train=load_significance_split(dataset_name, split_num, 'TRAIN')
        model=eval(model_name)(urm_train)
        model.fit(**params_dict)
        model.save_model(folder_path=save_model_base_path,
                         file_name='{}_{}'.format(model_name, split_num),
                         save_description=False)

        # two lines needed to free up RAM, since the similarities matrices are not freed up
        model=None
        gc.collect()


if __name__ == '__main__':
    action=menu.single_choice('what to do?', labels=[
                                'split', 'train_and_save'])
    if action == 'split':
        # dataset selection
        dataset_name=menu.single_choice('Select the dataset you want to create',
                                          ['Movielens1MReader', 'LastFMHetrec2011Reader', 'BookCrossingReader',
                                           'CiteULike_aReader', 'PinterestReader'])
        create_and_save_k_split()

    elif action == 'train_and_save':
        models=['BPR_NNMF', 'FUNK_NNMF', 'PROB_NNMF',
            'BPRMF', 'FunkSVD', 'PROBMF']
        datasets=['BookCrossing', 'CiteULike-a', 'LastFMHetrec2011',
                    'Movielens1M', 'Pinterest']

        selected_datasets=menu.options(
            options=datasets, labels=datasets, title='Select datasets', enable_all=True)
        selected_models=menu.options(
            options=models, labels=models, title='Select models', enable_all=True)

        for dataset in selected_datasets:
            for model in selected_models:
                _train_and_save_models(dataset_name=dataset, model_name=model)
