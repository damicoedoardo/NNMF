from RecSysFramework.Evaluation.Evaluator import EvaluatorMetrics
from RecSysFramework.Utils.get_holdout import retrieve_train_validation_test_holdhout_dataset
from RecSysFramework.Recommender.NonPersonalized import TopPop
from RecSysFramework.Experiments.utils_experiments import get_results_latex_from_dict, round_all_digits
from RecSysFramework.Utils.get_model_from_best_models import get_model
import matplotlib.pyplot as plt
from RecSysFramework.Evaluation.Evaluator import EvaluatorHoldout
from RecSysFramework.Utils.compute_popularity import *
import numpy as np

def comp_perc(mf, nnmf):
    return 100*(nnmf-mf)/min(mf, nnmf)

def get_result_list(urm_train, urm_test, cutoff_list, metric_list, models_objects, cut_list, disjoint=True):
    pop = compute_popularity_user(urm_train)
    users, interactions = zip(*pop)
    users = np.array(users)
    interactions = np.array(interactions)
    cum_sum_interactions = np.cumsum(interactions)
    tot_interactions = np.sum(interactions)
    evalh = EvaluatorHoldout(cutoff_list, metric_list, minRatingsPerUser=0)
    evalh.global_setup(urm_test)
    results_list = [evalh.evaluateRecommender(
        m).get_results_dictionary(per_user=True) for m in models_objects]
    for t in zip(cut_list, cut_list[1:]):
        if disjoint:
            users_to_cons = users[(cum_sum_interactions >= t[0]*tot_interactions) & (cum_sum_interactions <= t[1]*tot_interactions)]
        else:
            users_to_cons = users[cum_sum_interactions <= t[1]*tot_interactions]
        perc_bpr.append(comp_perc(np.average(np.array(results_list[0][5]['MAP'])[users_to_cons]), np.average(np.array(results_list[1][5]['MAP'])[users_to_cons])))
        perc_funk.append(comp_perc(np.average(np.array(results_list[2][5]['MAP'])[users_to_cons]), np.average(np.array(results_list[3][5]['MAP'])[users_to_cons])))
        perc_prob.append(comp_perc(np.average(np.array(results_list[4][5]['MAP'])[users_to_cons]), np.average(np.array(results_list[5][5]['MAP'])[users_to_cons])))
    return perc_bpr, perc_funk, perc_prob

cutoff_list = [5, ]
metrics_list = [EvaluatorMetrics.MAP,]
disjoint = False

l = ['LastFMHetrec2011Reader', 'Movielens1MReader', 'BookCrossingReader', 'CiteULike_aReader', 'PinterestReader']
for name in l:
    train, test, validation, dataset_name = retrieve_train_validation_test_holdhout_dataset(name)


    models_objects = []
    models_names = []
    m1, model_name = get_model(train, dataset_name, 'BPRMF')
    models_objects.append(m1)
    models_names.append(model_name)
    m1, model_name = get_model(train, dataset_name, 'BPR_NNMF')
    models_objects.append(m1)
    models_names.append(model_name)
    m1, model_name = get_model(train, dataset_name, 'FunkSVD')
    models_objects.append(m1)
    models_names.append(model_name)
    m1, model_name = get_model(train, dataset_name, 'FUNK_NNMF')
    models_objects.append(m1)
    models_names.append(model_name)
    m1, model_name = get_model(train, dataset_name, 'probmf')
    models_objects.append(m1)
    models_names.append(model_name)
    m1, model_name = get_model(train, dataset_name, 'nnprobmf')
    models_objects.append(m1)
    models_names.append(model_name)

    perc_bpr = []
    perc_funk = []
    perc_prob = []
    cut_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1]
    perc_bpr, perc_funk, perc_prob = get_result_list(train.get_URM(), validation.get_URM(), cutoff_list, metrics_list, models_objects, cut_list, disjoint=disjoint)

    plt.plot(cut_list[1:], perc_bpr, label='bpr')
    plt.plot(cut_list[1:], perc_funk, label='funk')
    plt.plot(cut_list[1:], perc_prob, label='prob')
    plt.axhline(y=0, color='gray', linestyle='--')
    axes = plt.gca()
    axes.set_ylim([-100,100])
    plt.legend()
    if disjoint:
        plt.savefig('{}_performance_varying_user_pop_disjoint.png'.format(dataset_name))
    else:
        plt.savefig('{}_performance_varying_user_pop_joint.png'.format(dataset_name))
    plt.close()
