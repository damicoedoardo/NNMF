from RecSysFramework.Utils.compute_popularity import compute_popularity_item, compute_popularity_user
import numpy as np
from RecSysFramework.Utils.get_holdout import retrieve_train_validation_test_holdhout_dataset
from RecSysFramework.Utils import menu
from RecSysFramework.Experiments.utils_experiments import get_results_latex_from_dict


def statistics_interations_per_user_long_queue_items(URM_train, URM_validation, thrs, eval_on_pop=False):
    d = {}
    items, interactions = zip(*compute_popularity_item(URM_train))
    items = np.array(items)
    interactions = np.array(interactions)
    cum_sum_interactions = np.cumsum(interactions)
    tot_interactions = np.sum(interactions)
    for thr in thrs:
        URM_copy = URM_validation.copy()
        items_to_exclude = items[cum_sum_interactions <= thr *
                                 tot_interactions] if eval_on_pop else items[cum_sum_interactions > thr*tot_interactions]

        for i in items_to_exclude:
            URM_copy.data[URM_copy.indices == i] = 0
            URM_copy.eliminate_zeros()

        _, ratings = zip(*compute_popularity_user(URM_copy))
        d[thr] = sum(ratings)/len(ratings)

    return d


def statistics_interations_per_user_long_queue_users(URM_train, URM_validation, thrs, eval_on_pop=False):
    d = {}

    users, interations_train = zip(*compute_popularity_user(URM_train))
    users = np.array(users)
    interations_train = np.array(interations_train)
    cum_sum_interations_train = np.cumsum(interations_train)
    tot_interations_train = np.sum(interations_train)

    pop = compute_popularity_user(URM_validation)
    pop = sorted(pop, key=lambda x: x[0])
    _, interations_validation = zip(*pop)
    interations_validation = np.array(interations_validation)
    for thr in thrs:
        users_in_cut = users[cum_sum_interations_train > thr *
                             tot_interations_train] if eval_on_pop else users[cum_sum_interations_train <= thr*tot_interations_train]

        interactions_on_validation_users_in_cut = interations_validation[users_in_cut]
        d[thr] = np.mean(interactions_on_validation_users_in_cut)

    return d


def gen_latex_code(d, tr='item'):
    s = '\\textbf{Cutoffs} '
    for thr in d.keys():
        s += '& {} '.format(thr)
    s += '\\\\ \\midrule \n Interactions per {} {} \\\\ \\bottomrule'.format(tr, get_results_latex_from_dict({'fake': d}, n_digits=4))
    return s

if __name__ == '__main__':
    thrs = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1]
    train, test, validation, dataset_name = retrieve_train_validation_test_holdhout_dataset()

    d_users = statistics_interations_per_user_long_queue_users(
        train.get_URM(), validation.get_URM(), thrs)
    print(gen_latex_code(d_users, 'user'))

    d_items = statistics_interations_per_user_long_queue_items(train.get_URM(), validation.get_URM(), thrs)
    print(gen_latex_code(d_items, 'item'))
