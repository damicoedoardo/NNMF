from RecSysFramework.Experiments.generate_performance_table import print_performance_table
from RecSysFramework.Utils.get_model_from_best_models import get_model
from RecSysFramework.Utils.get_holdout import retrieve_train_validation_test_holdhout_dataset
from RecSysFramework.Recommender.MatrixFactorization.PureSVD import PureSVD
from RecSysFramework.Recommender.SLIM.BPR.SLIM_BPR_Cython import SLIM_BPR_Cython
from RecSysFramework.Experiments.recommended_items_popularity_analysis import recommend_batch, rec_and_hit_items_pop_analysis
from RecSysFramework.Recommender.NonPersonalized import TopPop
import numpy as np

def run_on_single_dataset_lt_sh_table(train, test, validation, dataset_name):
    cutoff_list = [5, ]
    # train, test, validation, dataset_name = retrieve_train_validation_test_holdhout_dataset()

    top_pop = TopPop(train.get_URM())
    top_pop.fit()

    models_objects = [top_pop]
    models_names = ['TopPop']

    m1, model_name = get_model(train, dataset_name, 'ItemKNNCF')
    models_objects.append(m1)
    models_names.append(model_name)
    m1, model_name = get_model(train, dataset_name, 'UserKNNCF')
    models_objects.append(m1)
    models_names.append(model_name)
    if dataset_name != 'Movielens20M':
        m1, model_name = get_model(train, dataset_name, 'SLIM_BPR_Cython')
        models_objects.append(m1)
        models_names.append(model_name)
    m1, model_name = get_model(train, dataset_name, 'PureSVD')
    models_objects.append(m1)
    models_names.append(model_name)
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

    thrs = [0.66, ]
    for cutoff in cutoff_list:
        results_list = []
        for rec, name in zip(models_objects, models_names):
            recommendations = recommend_batch(
                rec, np.arange(rec.URM_train.shape[0]), cutoff=5)
            _, rec_and_hit = rec_and_hit_items_pop_analysis(
                train.get_URM(), validation.get_URM(), thrs, recommendations, )
            results_list.append({5: {'Hits long tail': int(rec_and_hit[0]), 'Hits short head': int(rec_and_hit[1]), 'Total': int(rec_and_hit[0]) + int(rec_and_hit[1])}})

        caption = 'Number of hits on popular and non popular items in the dataset ' + \
            dataset_name + ' with cutoff at ' + str(cutoff) + '. Bold indicates the highest number of hits. Underline indicates which algorithm performs an higher number of hits between MF and NNMF, pairwise.'
        label = 'n_hits_' + str(cutoff) + '_' + dataset_name
        ml = list(results_list[0][cutoff].keys())
        first_row = '\\multirow{2}{*}{\\textbf{Algorithm}} & \multicolumn{3}{c}{\\textbf{Hits}} \\\\ & Long tail & Short head & Total'
        print_performance_table(results_list, train, test, validation,
                                dataset_name, models_names, caption, label, cutoff, first_row, 'c|' + 'c'*len(ml))

if __name__ == '__main__':
    train, test, validation, dataset_name = retrieve_train_validation_test_holdhout_dataset()
    run_on_single_dataset_lt_sh_table(train, test, validation, dataset_name)
