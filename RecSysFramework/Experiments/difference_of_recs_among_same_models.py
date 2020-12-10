
from RecSysFramework.Recommender.MatrixFactorization.MatrixFactorization_Cython import BPR_NNMF, FUNK_NNMF, PROB_NNMF
from RecSysFramework.Evaluation.Comparator import ComparatorHoldoutUserPopularity
from RecSysFramework.DataManager.Splitter import Holdout
from RecSysFramework.Evaluation.Evaluator import EvaluatorHoldout
from RecSysFramework.Utils.WriteTextualFile import WriteTextualFile
from RecSysFramework.Utils.Common import avgDicts
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import copy
from RecSysFramework.Utils.compute_popularity import compute_popularity_item, compute_popularity_user
from RecSysFramework.Utils.get_holdout import retrieve_train_validation_test_holdhout_dataset
from RecSysFramework.Experiments.generate_performance_table import print_performance_table
from functools import partial
from tqdm import tqdm
import math
from RecSysFramework.Utils import menu


def compute_latent_nn_factor_in_batch(similarity, factor):
    # Utility method to compute the nearest neighbors of each user and item latent factor
    # reducing the memory consumption, without slowing down
    nn_factor = np.zeros(factor.shape)
    size = 1000
    n_rows = similarity.shape[0]

    n_batch = n_rows // size
    for idx in range(n_batch):
        nn_factor[size*idx: size *
                  (idx+1), :] = similarity[size*idx: size*(idx+1), :].dot(factor)
    nn_factor[(size*n_batch) % n_rows: n_rows,
              :] = similarity[(size*n_batch) % n_rows: n_rows, :].dot(factor)
    return nn_factor


def compute_jaccard(d1, d2):
    j = []
    for key, l1 in d1.items():
        s1 = set(l1)
        s2 = set(d2[key])
        j.append(len(s1 & s2)/len(s1 | s2))
    return sum(j)/len(j)


def latent_factors_difference(m1, m2, user_list=None, item_list=None, cuts=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1], eval_on_pop=False, eval_on_range=False, cutoffs=[10, 25, 50, 100, 200]):
    # given two models, compute the degree of similarity of their latent factor representation,
    # carrying out the Stability of Representation experiments

    assert not (
        eval_on_pop and eval_on_range), "Pick one among eval on top and eval on range"

    # Compute the neighborhood aware embeddings. In case I'm evaluating a MF,
    # the User or Item similarity is the identity matrix, so the neighborhood aware
    latent_factor_user_m1 = compute_latent_nn_factor_in_batch(
        m1.user_similarity, m1.USER_factors)
    latent_factor_item_m1 = compute_latent_nn_factor_in_batch(
        m1.item_similarity, m1.ITEM_factors)
    latent_factor_user_m2 = compute_latent_nn_factor_in_batch(
        m2.user_similarity, m2.USER_factors)
    latent_factor_item_m2 = compute_latent_nn_factor_in_batch(
        m2.item_similarity, m2.ITEM_factors)

    # Compute the NN of each user and item latent factor
    norm_m1 = np.linalg.norm(latent_factor_item_m1, axis=1)
    norm_m2 = np.linalg.norm(latent_factor_item_m2, axis=1)
    norm_m1[norm_m1 == 0.0] = 1.0
    norm_m2[norm_m2 == 0.0] = 1.0
    P_m1 = latent_factor_item_m1/norm_m1[:, None]
    P_m2 = latent_factor_item_m2/norm_m2[:, None]
    d_m1_item = {}
    d_m2_item = {}
    for i in tqdm(range(latent_factor_item_m1.shape[0])):
        cosine_i_m1 = np.dot(P_m1[i], P_m1.T)
        cosine_i_m2 = np.dot(P_m2[i], P_m2.T)
        cosine_i_m1[i] = -np.inf
        cosine_i_m2[i] = -np.inf
        ind_m1 = np.argpartition(cosine_i_m1, -max(cutoffs))[-max(cutoffs):]
        ind_m2 = np.argpartition(cosine_i_m2, -max(cutoffs))[-max(cutoffs):]
        indices_neighbors_i_m1 = ind_m1[np.argsort(-cosine_i_m1[ind_m1])]
        indices_neighbors_i_m2 = ind_m2[np.argsort(-cosine_i_m2[ind_m2])]
        d_m1_item[i] = indices_neighbors_i_m1.astype(np.uint32)
        d_m2_item[i] = indices_neighbors_i_m2.astype(np.uint32)

    norm_m1 = np.linalg.norm(latent_factor_user_m1, axis=1)
    norm_m2 = np.linalg.norm(latent_factor_user_m2, axis=1)
    norm_m1[norm_m1 == 0.0] = 1.0
    norm_m2[norm_m2 == 0.0] = 1.0
    P_m1 = latent_factor_user_m1/norm_m1[:, None]
    P_m2 = latent_factor_user_m2/norm_m2[:, None]
    d_m1_user = {}
    d_m2_user = {}
    for i in tqdm(range(latent_factor_user_m1.shape[0])):
        cosine_i_m1 = np.dot(P_m1[i], P_m1.T)
        cosine_i_m2 = np.dot(P_m2[i], P_m2.T)
        cosine_i_m1[i] = -np.inf
        cosine_i_m2[i] = -np.inf
        ind_m1 = np.argpartition(cosine_i_m1, -max(cutoffs))[-max(cutoffs):]
        ind_m2 = np.argpartition(cosine_i_m2, -max(cutoffs))[-max(cutoffs):]
        indices_neighbors_i_m1 = ind_m1[np.argsort(-cosine_i_m1[ind_m1])]
        indices_neighbors_i_m2 = ind_m2[np.argsort(-cosine_i_m2[ind_m2])]
        d_m1_user[i] = indices_neighbors_i_m1.astype(np.uint32)
        d_m2_user[i] = indices_neighbors_i_m2.astype(np.uint32)

    results = {}

    # Compute user and item popularity, as we were interested on evaluating the
    # Jaccard index distinguishing users and items according to their popularity
    pop = compute_popularity_user(m1.URM_train)
    users, interactions_user = zip(*pop)
    users = np.array(users)
    interactions_user = np.array(interactions_user)
    cum_sum_interactions_user = np.cumsum(interactions_user)
    tot_interactions_user = np.sum(interactions_user)

    pop = compute_popularity_item(m1.URM_train)
    items, interactions_item = zip(*pop)
    items = np.array(items)
    interactions_item = np.array(interactions_item)
    cum_sum_interactions_item = np.cumsum(interactions_item)
    tot_interactions_item = np.sum(interactions_item)

    if not eval_on_range:
        for thr in cuts:

            if eval_on_pop:
                items_to_cons = items[cum_sum_interactions_item >=
                                      thr*tot_interactions_item]
            else:
                items_to_cons = items[cum_sum_interactions_item <=
                                      thr*tot_interactions_item]
            for cutoff in cutoffs:
                d_m1_item_cut_cutoff = {
                    k: d_m1_item[k][:cutoff] for k in items_to_cons}
                d_m2_item_cut_cutoff = {
                    k: d_m2_item[k][:cutoff] for k in items_to_cons}
                results['jaccard_{}_{}_Jaccard items'.format(thr, cutoff)] = compute_jaccard(
                    d_m1_item_cut_cutoff, d_m2_item_cut_cutoff)

            if eval_on_pop:
                users_to_cons = users[cum_sum_interactions_user >=
                                      thr*tot_interactions_user]
            else:
                users_to_cons = users[cum_sum_interactions_user <=
                                      thr*tot_interactions_user]
            for cutoff in cutoffs:
                d_m1_user_cut_cutoff = {
                    k: d_m1_user[k][:cutoff] for k in users_to_cons}
                d_m2_user_cut_cutoff = {
                    k: d_m2_user[k][:cutoff] for k in users_to_cons}
                results['jaccard_{}_{}_Jaccard users'.format(thr, cutoff)] = compute_jaccard(
                    d_m1_user_cut_cutoff, d_m2_user_cut_cutoff)

    else:
        for thr in zip(cuts, cuts[1:]):

            items_to_cons = items[(cum_sum_interactions_item <= thr[0]*tot_interactions_item) &
                                  (cum_sum_interactions_item > thr[1]*tot_interactions_item)]
            for cutoff in cutoffs:
                d_m1_item_cut_cutoff = {
                    k: d_m1_item[k][:cutoff] for k in items_to_cons}
                d_m2_item_cut_cutoff = {
                    k: d_m2_item[k][:cutoff] for k in items_to_cons}
                results['jaccard_{}-{}_{}_Jaccard items'.format(thr[0], thr[1], cutoff)] = compute_jaccard(
                    d_m1_item_cut_cutoff, d_m2_item_cut_cutoff)

    return results


def compute_recommendations_similarity(model, params_dict, train, test, validation, dataset_name, alg='MF', n_trials=10, cuts=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1],
                                       cutoffs_RBO=[10, 25, 50, 100], cutoffs_jaccard=[1, 3, 5, 10], choice='both', exp_kind='Comparable number of updates (procedure as explained in the paper)',
                                       cutoffs_representations=[10, 25, 50, 100, 200], eval_on_pop=False, eval_on_range=False):
    # given two models, compute the degree of similarity of recommendation,
    # carrying out the Stability of Representation experiments
    # Inside of this method we also call the Stability of Recommendation method

    result_dicts_jaccard = []
    result_dicts_RBO = []
    result_dicts_diff_latent_factors = []
    results_dict = {}
    result_dict_diff_latent_factors = {}

    # Load cached model, if exist. Otherwise fit and save it
    model_path = 'StabilityExperimentsModels/{}/{}/'.format(
        dataset_name, model.RECOMMENDER_NAME)
    if exp_kind == 'Optimal parameters' and alg == 'NNMF':
        file_name = '{}_{}_opt'.format(alg, 0)
    else:
        file_name = '{}_{}'.format(alg, 0)
    full_path = os.path.join(model_path, file_name)
    if os.path.exists(full_path + '.zip'):
        print('using cached model, {}'.format(full_path))
        model.load_model(folder_path=model_path, file_name=file_name)
    else:
        model.fit(**params_dict)
        model.save_model(folder_path=model_path, file_name=file_name)

    for idx in range(1, n_trials):
        if model.RECOMMENDER_NAME == 'BPR_NNMF':
            clone_model = BPR_NNMF(train.get_URM())
        elif model.RECOMMENDER_NAME == 'FUNK_NNMF':
            clone_model = FUNK_NNMF(train.get_URM())
        elif model.RECOMMENDER_NAME == 'PROB_NNMF':
            clone_model = PROB_NNMF(train.get_URM())

        # Load cached model, if exist. Otherwise fit and save it
        if exp_kind == 'Optimal parameters' and alg == 'NNMF':
            file_name = '{}_{}_opt'.format(alg, idx)
        else:
            file_name = '{}_{}'.format(alg, idx)
        full_path = os.path.join(model_path, file_name)
        if os.path.exists(full_path + '.zip'):
            print('using cached model, {}'.format(full_path))
            clone_model.load_model(folder_path=model_path, file_name=file_name)
        else:
            params_dict['random_seed'] = idx
            clone_model.fit(**params_dict)
            clone_model.save_model(folder_path=model_path, file_name=file_name)

        if choice == 'representations' or choice == 'both':
            result_dicts_diff_latent_factors.append(
                latent_factors_difference(model, clone_model, cuts=cuts, cutoffs=cutoffs_representations, eval_on_pop=eval_on_pop, eval_on_range=eval_on_range))

        if choice == 'recommendations' or choice == 'both':
            print('computing RBO for {}'.format(idx))
            d_to_add_RBO = {}
            for cutoff in cutoffs_RBO:
                comparator = ComparatorHoldoutUserPopularity(train.get_URM(),
                                                             validation.get_URM(),
                                                             [model, clone_model],
                                                             cutoff=cutoff,
                                                             cuts=cuts,
                                                             save_to_file=False,
                                                             verbose=False,
                                                             metrics_list=[
                    'RBO']
                )
                _, d = comparator.compare()
                d_to_add_RBO = {**d_to_add_RBO, **d}
            result_dicts_RBO.append(d_to_add_RBO)

            print('computing jaccard for {}'.format(idx))
            d_to_add_jaccard = {}
            for cutoff in cutoffs_jaccard:
                comparator = ComparatorHoldoutUserPopularity(train.get_URM(),
                                                             validation.get_URM(),
                                                             [model, clone_model],
                                                             cutoff=cutoff,
                                                             cuts=cuts,
                                                             save_to_file=False,
                                                             verbose=False,
                                                             metrics_list=[
                    'jaccard']
                )
                _, d = comparator.compare()
                d_to_add_jaccard = {**d_to_add_jaccard, **d}
            result_dicts_jaccard.append(d_to_add_jaccard)

    if choice == 'recommendations' or choice == 'both':
        results_dict['RBO'] = avgDicts(result_dicts_RBO)
        results_dict['jaccard'] = avgDicts(result_dicts_jaccard)
    if choice == 'representations' or choice == 'both':
        result_dict_diff_latent_factors = avgDicts(
            result_dicts_diff_latent_factors)

    return results_dict, result_dict_diff_latent_factors


def _read_params_file(dataset_name, model_name):
    # Open the optimal parameters file in the folder BestModels

    filenames = np.array(os.listdir(
        'BestModels/{}/{}'.format(dataset_name, model_name)))
    # mantain only the filename with .txt in the file
    mask = np.array(['.txt' in fn for fn in filenames])
    filename = filenames[mask][0]  # .split('.txt')[0]
    params_file = open(
        'BestModels/{}/{}/{}'.format(dataset_name, model_name, filename), 'r')
    return params_file.read()


def get_param(params_string, param_name):
    if param_name == 'latent_factors_initialization':
        return eval(params_string.split(param_name + '=')[1].split('}')[0] + '}')
    elif param_name == 'epochs':
        return int(params_string.split('epochs=')[-1].split(',')[0])
    else:
        return eval(params_string.split(param_name + '=')[1].split(',')[0])


def get_params_dict_model_dataset(dataset_name, model_name, evalh):
    params_string = _read_params_file(dataset_name, model_name)
    _get_param = partial(get_param, params_string)
    if model_name == 'BPRMF' or model_name == 'BPR_NNMF':
        epochs = _get_param('epochs')
        params_dict = {
            'item_k': 1 if model_name == 'BPRMF' else _get_param('item_k'),
            'user_k': 1 if model_name == 'BPRMF' else _get_param('user_k'),
            'num_factors': _get_param('num_factors'),
            'learning_rate': _get_param('learning_rate'),
            'item_reg': _get_param('positive_reg') if model_name == 'BPRMF' else _get_param('item_reg'),
            'user_reg': _get_param('user_reg'),
            'positive_reg': _get_param('positive_reg'),
            'negative_reg': _get_param('negative_reg'),
            'latent_factors_initialization': _get_param('latent_factors_initialization'),
            'validation_every_n': epochs,
            'sgd_mode': 'adam',
            'lower_validations_allowed': 50,
            'validation_metric': 'MAP',
            'evaluator_object': evalh,
            'stop_on_validation': True,
            'verbose': False,
            'use_bias': False,
            'positive_threshold_BPR': 0,
            'epochs': epochs,
            'item_shrink': 0 if model_name == 'BPRMF' else _get_param('item_shrink'),
            'user_shrink': 0 if model_name == 'BPRMF' else _get_param('user_shrink'),
            'batch_size': _get_param('batch_size'),
            'random_seed': 0,
            'user_eye': False,
            'item_eye': False,
        }

    elif model_name == 'FunkSVD' or model_name == 'FUNK_NNMF':
        epochs = _get_param('epochs')
        params_dict = {
            'item_k': 1 if model_name == 'FunkSVD' else _get_param('item_k'),
            'user_k': 1 if model_name == 'FunkSVD' else _get_param('user_k'),
            'num_factors': _get_param('num_factors'),
            'learning_rate': _get_param('learning_rate'),
            'item_reg': _get_param('item_reg'),
            'user_reg': _get_param('user_reg'),
            'negative_interactions_quota': _get_param('negative_interactions_quota'),
            'latent_factors_initialization': _get_param('latent_factors_initialization'),
            'validation_every_n': epochs,
            'sgd_mode': 'adam',
            'lower_validations_allowed': 50,
            'validation_metric': 'MAP',
            'evaluator_object': evalh,
            'stop_on_validation': True,
            'verbose': False,
            'use_bias': False,
            'positive_threshold_BPR': 0,
            'epochs': epochs,
            'item_shrink': 0 if model_name == 'FunkSVD' else _get_param('item_shrink'),
            'user_shrink': 0 if model_name == 'FunkSVD' else _get_param('user_shrink'),
            'batch_size': _get_param('batch_size'),
            'random_seed': 0,
            'user_eye': False,
            'item_eye': False,
        }

    elif model_name == 'probmf' or model_name == 'nnprobmf':
        epochs = _get_param('epochs')
        params_dict = {
            'item_k': 1 if model_name == 'probmf' else _get_param('item_k'),
            'user_k': 1 if model_name == 'probmf' else _get_param('user_k'),
            'num_factors': _get_param('num_factors'),
            'learning_rate': _get_param('learning_rate'),
            'item_reg': _get_param('item_reg'),
            'user_reg': _get_param('user_reg'),
            'latent_factors_initialization': _get_param('latent_factors_initialization'),
            'validation_every_n': epochs,
            'sgd_mode': 'adam',
            'lower_validations_allowed': 50,
            'validation_metric': 'MAP',
            'evaluator_object': evalh,
            'stop_on_validation': True,
            'verbose': False,
            'use_bias': False,
            'positive_threshold_BPR': 0,
            'epochs': epochs,
            'item_shrink': 0 if model_name == 'probmf' else _get_param('item_shrink'),
            'user_shrink': 0 if model_name == 'probmf' else _get_param('user_shrink'),
            'batch_size': _get_param('batch_size'),
            'random_seed': 0,
            'user_eye': False,
            'item_eye': False,
        }

    return params_dict


def difference_of_recs_among_same_models():
    exp_kind = 'Optimal parameters'
    choice = menu.single_choice(
        'what want to do', ['recommendations', 'representations', 'both', 'none'])

    models_names = menu.options(['BPRMF', 'FunkSVD', 'probmf'], [
        'BPR', 'Funk', 'Prob'])

    train, test, validation, dataset_name = retrieve_train_validation_test_holdhout_dataset()

    # Evaluator is not really used in this test
    evalh = EvaluatorHoldout([5])
    evalh.global_setup(URM_test=test.get_URM())

    results_list_recommendation = []
    results_list_representations = []
    model_names = []

    for model_name in models_names:
        nn_version_name = model_name.replace('probmf', 'nnprobmf').replace(
            'FunkSVD', 'FUNKMF').replace('MF', '_NNMF')
        if model_name == 'BPRMF':
            algorithm = BPR_NNMF(train.get_URM())
        elif model_name == 'FunkSVD':
            algorithm = FUNK_NNMF(train.get_URM())
        elif model_name == 'probmf':
            algorithm = PROB_NNMF(train.get_URM())
        params_dict = get_params_dict_model_dataset(
            dataset_name, model_name, evalh)

        # compute stability of recommendations and representations of basic MF
        metric_fst, repr_fst = compute_recommendations_similarity(
            algorithm, params_dict, train, test, validation, dataset_name, alg='MF', n_trials=10, cutoffs_RBO=[5, 10, 25, ],
            cutoffs_jaccard=[1, 5, 10, 25], cuts=[1], choice=choice, exp_kind=exp_kind)

        if exp_kind == 'Comparable number of updates (procedure as explained in the paper)':
            params_dict['item_k'] = 5
            params_dict['user_k'] = 5
            params_dict['epochs'] = math.floor(params_dict['epochs']/5)
            params_dict['validation_every_n'] = math.floor(
                params_dict['validation_every_n']/5)
        elif exp_kind == 'Optimal parameters':
            params_dict = get_params_dict_model_dataset(
                dataset_name, nn_version_name, evalh)

        # compute stability of recommendations and representations of NNMF
        metric_snd, repr_snd = compute_recommendations_similarity(
            algorithm, params_dict, train, test, validation, dataset_name, alg='NNMF', n_trials=10, cutoffs_RBO=[5, 10, 25, ],
            cutoffs_jaccard=[1, 5, 10, 25], cuts=[1], choice=choice, exp_kind=exp_kind)

        model_name = model_name.replace('probmf', 'ProbMF')
        model_names.append(model_name)
        model_names.append(model_name.replace(
            'MF', ' NNMF').replace('SVD', ' NNMF'))
        if choice == 'recommendations' or choice == 'both':
            results_list_recommendation.append(
                {0: {**metric_fst['jaccard'], **metric_fst['RBO']}})
            results_list_recommendation.append(
                {0: {**metric_snd['jaccard'], **metric_snd['RBO']}})
        if choice == 'representations' or choice == 'both':
            results_list_representations.append({0: repr_fst})
            results_list_representations.append({0: repr_snd})

    # generate results tables, starting from values of metrics
    if choice == 'recommendations' or choice == 'both':
        first_row = '\\multirow{2}{*}{\\textbf{Algorithm}} & \multicolumn{4}{c|}{\\textbf{Jaccard}} & \multicolumn{3}{c}{\\textbf{RBO}} \\\\ & 1 & 5 & 10 & 25 & 5 & 10 & 25'
        print_performance_table(results_list_recommendation, train.get_URM(), test.get_URM(), validation.get_URM(),
                                dataset_name, model_names, 'Stability of recommendations on the dataset {}'.format(dataset_name), 'stability_recommendations_{}'.format(dataset_name), 0, first_row, 'c|cccc|ccc')
    if choice == 'representations' or choice == 'both':
        first_row = '\\multirow{2}{*}{\\textbf{Algorithm}} & \\multicolumn{5}{c|}{\\textbf{Jaccard items}} & \\multicolumn{5}{c}{\\textbf{Jaccard users}} \\\\ & 10 & 25 & 50 & 100 & 200 & 10 & 25 & 50 & 100 & 200'
        print_performance_table(results_list_representations, train.get_URM(), test.get_URM(), validation.get_URM(),
                                dataset_name, model_names, 'Stability of representations on the dataset {}'.format(dataset_name), 'stability_representations_{}'.format(dataset_name), 0, first_row, 'c|ccccc|ccccc', higher_better=True)


if __name__ == '__main__':
    difference_of_recs_among_same_models()
