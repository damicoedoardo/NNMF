import numpy as np
from RecSysFramework.Recommender.MatrixFactorization.MatrixFactorization_Cython import BPR_NNMF, BPRMF, FUNK_NNMF, FunkSVD
from RecSysFramework.Recommender.MatrixFactorization.PureSVD import PureSVD
from RecSysFramework.Recommender.KNN.ItemKNNCFRecommender import ItemKNNCF
from RecSysFramework.Recommender.KNN.UserKNNCFRecommender import UserKNNCF
from RecSysFramework.DataManager.Splitter import Holdout
from RecSysFramework.Evaluation.Evaluator import EvaluatorHoldout
from RecSysFramework.DataManager.DatasetPostprocessing.ImplicitURM import ImplicitURM
from RecSysFramework.DataManager.DatasetPostprocessing.KCore import KCore
from tqdm import tqdm
import RecSysFramework.Utils.compute_popularity as cp
from RecSysFramework.Utils.get_holdout import retrieve_train_validation_test_holdhout_dataset
import pandas as pd
import matplotlib.pyplot as plt
from RecSysFramework.Utils.get_model_from_best_models import get_model
from RecSysFramework.Utils import menu
from collections import OrderedDict
from RecSysFramework.Experiments.utils_experiments import round_all_digits


def recommend_batch(recommender_object, users, remove_seen_flag=True, cutoff=5, remove_top_pop_flag=False):
    recs = []
    size = 1000
    n_users = len(users)

    n_batch = n_users // size
    for idx in range(n_batch):
        recs += recommender_object.recommend(
            users[size * idx: size *
                  (idx + 1)], remove_seen_flag=remove_seen_flag, cutoff=cutoff,
            remove_top_pop_flag=remove_top_pop_flag, return_scores=True)[0]
    recs += recommender_object.recommend(
        users[(size * n_batch) %
              n_users: n_users], remove_seen_flag=remove_seen_flag, cutoff=cutoff,
        remove_top_pop_flag=remove_top_pop_flag, return_scores=True)[0]
    return np.array(recs)


def rec_items_pop_analysis(urm_train, cut_list, recs):
    """
    plot the percentage of items recommended in every popularity section
    """

    # initialize an empty vector of equal length of the items in the dataset
    # we will accumulate on this the number of times that the recommender recommend an
    # item
    items_predictions_num = np.zeros(urm_train.shape[1])

    for user_recs in tqdm(recs):
        items_predictions_num[user_recs] += 1

    item_pop_tuple_list = cp.compute_popularity_item(urm_train)
    items_idxs, interactions = zip(*item_pop_tuple_list)

    interactions_cumsum = np.cumsum(interactions)
    interactions_cumsum_norm = interactions_cumsum/max(interactions_cumsum)

    cut_idxs = []
    for cut_perc in cut_list:
        cut_idx = (np.abs(interactions_cumsum_norm - cut_perc)).argmin()
        cut_idxs.append(cut_idx)

    items_partition = np.split(items_idxs, cut_idxs)
    items_prediction_cuts = []

    for partition in items_partition:
        items_prediction_cuts.append(np.sum(items_predictions_num[partition]))

    items_prediction_cuts_norm = items_prediction_cuts / \
        np.sum(items_prediction_cuts)
    print(items_prediction_cuts_norm)
    return items_prediction_cuts_norm


def rec_and_hit_items_pop_analysis(urm_train, urm_validation, cut_list, recs):
    """
    plot the percentage of items recommended and hit in every popularity section
    """

    # initialize an empty vector of equal length of the items in the dataset
    # we will accumulate on this the number of times that the recommender recommend an
    # item and hit it
    items_predictions_num = np.zeros(urm_train.shape[1])

    for idx, user_recs in enumerate(recs):
        items_to_be_hit_user = urm_validation.indices[urm_validation.indptr[idx]
            :urm_validation.indptr[idx+1]]
        user_recs_hit = list(set(items_to_be_hit_user) & set(user_recs))
        if len(user_recs_hit) > 0:
            items_predictions_num[np.array(user_recs_hit, dtype=np.int)] += 1

    item_pop_tuple_list = cp.compute_popularity_item(urm_train)
    items_idxs, interactions = zip(*item_pop_tuple_list)

    interactions_cumsum = np.cumsum(interactions)
    interactions_cumsum_norm = interactions_cumsum/max(interactions_cumsum)

    cut_idxs = []
    for cut_perc in cut_list:
        cut_idx = (np.abs(interactions_cumsum_norm - cut_perc)).argmin()
        cut_idxs.append(cut_idx)

    items_partition = np.split(items_idxs, cut_idxs)
    items_prediction_cuts = []

    for partition in items_partition:
        items_prediction_cuts.append(np.sum(items_predictions_num[partition]))

    items_prediction_cuts_norm = items_prediction_cuts / \
        np.sum(items_prediction_cuts)
    print(items_prediction_cuts_norm)
    return items_prediction_cuts_norm, np.array(items_prediction_cuts)


def models_comparison_items_pop_analysis(recs, names, thrs, urm_validation, dataset_name):
    s = ''
    pop_recs = []
    pop_recs_hit_perc = []
    pop_recs_hit = []
    for rec, name in zip(recs, names):
        recommendations = recommend_batch(
            rec, np.arange(rec.URM_train.shape[0]))
        pop_recs.append(rec_items_pop_analysis(
            rec.URM_train, thrs, recommendations, ))

        rec_and_hit_perc, rec_and_hit = rec_and_hit_items_pop_analysis(
            rec.URM_train, urm_validation, thrs, recommendations, )
        pop_recs_hit_perc.append(rec_and_hit_perc)
        pop_recs_hit.append(rec_and_hit)

    file_name = 'predicted_items_pop_{}'.format(
        dataset_name).replace(' ', '_')
    analysis_df = pd.DataFrame({name: pop*100 for name, pop in zip(names, pop_recs)},
                               index=['Long tail', 'Short head'])
    ax = analysis_df.T.plot.barh(stacked=True, rot=0)
    ax.set_xlabel('Percentage')
    ax.get_yticklabels()[0].set_fontweight('bold')
    ax.get_yticklabels()[2].set_fontweight('bold')
    ax.get_yticklabels()[4].set_fontweight('bold')
    ax.margins(x=0)
    for idx, p in enumerate(ax.patches):
        x, y = p.get_xy()
        if x == 0:
            if idx in [0, 2, 4]:
                width = p.get_width()
                ax.text(7+p.get_width(), p.get_y()+0.5*p.get_height(),
                        '{:1.1f}%'.format(width),
                        ha='center', va='center', fontweight='bold',)
            else:
                width = p.get_width()
                ax.text(7+p.get_width(), p.get_y()+0.5*p.get_height(),
                        '{:1.1f}%'.format(width),
                        ha='center', va='center', )

    file_name = 'distr_recs_hits_{}'.format(
        dataset_name).replace(' ', '_')
    s += gen_latex_code('Distribution of popular and not popular items recommended by the algorithms. Bold highlights NNMFs. The dataset is: ' +
                        dataset_name + '.', file_name) + '.\n'

    plt.savefig(file_name, bbox_inches='tight')
    return s


def gen_latex_code(caption, file_name):
    return '\\begin{figure}[H] \n \
         \\includegraphics[width=1\\textwidth]{pictures/' + file_name + '.png} \n \
         \\centering \n \
         \\caption{' + caption + '} \n \
         \\label{fig:' + file_name + '} \n \
         \end{figure}'


def _recommended_items_popularity_analysis(thrs, train, test, validation, dataset_name, ):
    """runs the analysis on popularity bins GIVEN A DATASET
    moreover, saves the images and generate latex code that allows to
    import the image directly in latex
    """

    m1, m1_name = get_model(train, dataset_name, model_name='BPRMF')
    m2, m2_name = get_model(train, dataset_name, model_name='BPR_NNMF')
    m3, m3_name = get_model(train, dataset_name, model_name='FunkSVD')
    m4, m4_name = get_model(train, dataset_name, model_name='FUNK_NNMF')
    m5, m5_name = get_model(train, dataset_name, model_name='probmf')
    m6, m6_name = get_model(train, dataset_name, model_name='nnprobmf')
    m7, m7_name = get_model(train, dataset_name, model_name='ItemKNNCF')
    m8, m8_name = get_model(train, dataset_name, model_name='UserKNNCF')
    m9, m9_name = get_model(train, dataset_name, model_name='PureSVD')
    if dataset_name != 'Movielens20M':
        m10, m10_name = get_model(
            train, dataset_name, model_name='SLIM_BPR_Cython')
        s = models_comparison_items_pop_analysis(
            [m6, m5, m4, m3, m2, m1, m9, m10, m8, m7], [m6_name, m5_name, m4_name, m3_name, m2_name, m1_name, m9_name, m10_name, m8_name, m7_name], thrs, validation.get_URM(), dataset_name)
    else:
        s = models_comparison_items_pop_analysis(
            [m6, m5, m4, m3, m2, m1, m9, m8, m7], [m6_name, m5_name, m4_name, m3_name, m2_name, m1_name, m9_name, m8_name, m7_name], thrs, validation.get_URM(), dataset_name)

    print(s)


def recommended_items_popularity_analysis():
    thrs = [0.66, ]
    train, test, validation, dataset_name = retrieve_train_validation_test_holdhout_dataset()
    _recommended_items_popularity_analysis(
        thrs, train, test, validation, dataset_name, )


if __name__ == '__main__':
    recommended_items_popularity_analysis()
