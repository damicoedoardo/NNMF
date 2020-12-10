from RecSysFramework.Experiments.generate_performance_table import *
from RecSysFramework.Evaluation.Evaluator import EvaluatorMetrics
from RecSysFramework.Utils.get_holdout import retrieve_train_validation_test_holdhout_dataset
from RecSysFramework.Recommender.NonPersonalized import TopPop
from RecSysFramework.Experiments.utils_experiments import get_results_latex_from_dict, round_all_digits
from RecSysFramework.Utils.get_model_from_best_models import get_model
import matplotlib.pyplot as plt

def comp_perc(mf, nnmf):
    return 100*(nnmf-mf)/min(mf, nnmf)

cutoff_list = [5, ]
metrics_list = [EvaluatorMetrics.MAP,]

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
    cut_list = [1, 0.8, 0.66, 0.5, 0.4, 0.33, 0.2]
    for cut in cut_list:
        results_list = get_result_list(eliminate_item_on_short_head_from_test(train.get_URM(), validation.get_URM(), cut), cutoff_list, metrics_list, models_objects)
        perc_bpr.append(comp_perc(results_list[0][5]['MAP'], results_list[1][5]['MAP']))
        perc_funk.append(comp_perc(results_list[2][5]['MAP'], results_list[3][5]['MAP']))
        perc_prob.append(comp_perc(results_list[4][5]['MAP'], results_list[5][5]['MAP']))

    plt.plot(cut_list, perc_bpr, label='bpr')
    plt.plot(cut_list, perc_funk, label='funk')
    plt.plot(cut_list, perc_prob, label='prob')
    plt.axhline(y=0, color='gray', linestyle='--')
    axes = plt.gca()
    axes.set_ylim([-100,100])
    plt.legend()
    plt.savefig('{}_performance_varying_item_pop.png'.format(dataset_name))
    plt.close()
