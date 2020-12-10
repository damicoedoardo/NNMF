
from RecSysFramework.Evaluation.Evaluator import EvaluatorMetrics
from RecSysFramework.Recommender.MatrixFactorization.MatrixFactorization_Cython import BPRMF, BPR_NNMF, FunkSVD, FUNK_NNMF
from RecSysFramework.Recommender.KNN.ItemKNNCFRecommender import ItemKNNCF
from RecSysFramework.Recommender.KNN.UserKNNCFRecommender import UserKNNCF
from RecSysFramework.Recommender.MatrixFactorization.PureSVD import PureSVD
from RecSysFramework.Recommender.SLIM.BPR.SLIM_BPR_Cython import SLIM_BPR_Cython
from RecSysFramework.Utils.get_holdout import retrieve_train_validation_test_holdhout_dataset
from RecSysFramework.Utils import menu
from RecSysFramework.Evaluation.Evaluator import EvaluatorHoldout
from RecSysFramework.Recommender.NonPersonalized import TopPop
from RecSysFramework.Experiments.utils_experiments import get_results_latex_from_dict, round_all_digits
from RecSysFramework.Utils.get_model_from_best_models import get_model
import numpy as np
from RecSysFramework.Experiments.utils_experiments import get_items_long_tail_short_head


def print_performance_table(results_list, train, test, validation, dataset_name, models_names, caption, label, cutoff, first_row, table_format, digits=4, higher_better=True):
    s = ''
    s += '\\begin{table}[H] \\normalsize \n \
            \\centering \n \
            \\caption{' + caption + '} \n'
    s += '\\begin{tabular}{' + table_format + '} \n \
          \\toprule \n'

    s += first_row + '\\\\'

    best_results_dict = {}
    for metric in results_list[0][cutoff].keys():
        max_value = -np.inf if higher_better else np.inf
        for result in results_list:
            if higher_better:
                max_value = result[cutoff][metric] if result[cutoff][metric] > max_value else max_value
            else:
                max_value = result[cutoff][metric] if result[cutoff][metric] < max_value else max_value
        best_results_dict[metric] = max_value

    abs_dict = dict(zip(models_names, results_list))

    for d, name in zip(results_list, models_names):
        if name == 'BPRMF' or name == 'FunkSVD' or name == 'ProbMF' or name == models_names[0]:
            s += '\\midrule \n'
        else:
            s += ' \n'
        s += name + ' '

        for metric in d[cutoff].keys():
            prefix = ''
            suffix = ''
            if round(d[cutoff][metric], digits) == round(best_results_dict[metric], digits):
                prefix += '\\textbf{'
                suffix += '}'
            if name in ['BPRMF', 'FunkSVD', 'ProbMF']:
                nnmfname = name.replace('MF', ' NNMF').replace('SVD', ' NNMF')
                if d[cutoff][metric] > abs_dict[nnmfname][cutoff][metric] and higher_better:
                    prefix += '\\underline{'
                    suffix += '}'
                if d[cutoff][metric] < abs_dict[nnmfname][cutoff][metric] and not higher_better:
                    prefix += '\\underline{'
                    suffix += '}'
            if name in ['BPR NNMF', 'Funk NNMF', 'Prob NNMF']:
                mfname = name.replace('BPR NNMF', 'BPRMF').replace(
                    'Funk NNMF', 'FunkSVD').replace('Prob NNMF', 'ProbMF')
                if d[cutoff][metric] > abs_dict[mfname][cutoff][metric] and higher_better:
                    prefix += '\\underline{'
                    suffix += '}'
                if d[cutoff][metric] < abs_dict[mfname][cutoff][metric] and not higher_better:
                    prefix += '\\underline{'
                    suffix += '}'
            s += '& {}{}{} '.format(prefix,
                                    str(round_all_digits(d[cutoff][metric], digits)), suffix)
        s += '\\\\ '

    s += '\\bottomrule \n \\end{tabular} \n \\label{table:' + \
        label + '} \n \\end{table}'
    s += '\n\n'

    print(s)


def eliminate_item_on_short_head_from_test(urm_train, urm_test, cut_perc=0.66):
    _, short_head = get_items_long_tail_short_head(urm_train, cut_perc)

    for idx in short_head:
        urm_test.data[urm_test.indices == idx] = 0
    urm_test.eliminate_zeros()
    return urm_test


def get_result_list(urm_test, cutoff_list, metric_list, models_objects):
    evalh = EvaluatorHoldout(cutoff_list, metric_list,)
    evalh.global_setup(urm_test)
    results_list = [evalh.evaluateRecommender(
        m).get_results_dictionary() for m in models_objects]
    return results_list


def run_on_single_dataset_perf_table(train, test, validation, dataset_name, cutoff_list=[5, ], metrics_list=[EvaluatorMetrics.RECALL, EvaluatorMetrics.MAP],
                                     cut_percs=[0.66], prin=True):

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

    for cut_perc in cut_percs:
        results_list = get_result_list(eliminate_item_on_short_head_from_test(train.get_URM(
        ), test.get_URM(), cut_perc=cut_perc), cutoff_list, metrics_list, models_objects,)
        for cutoff in cutoff_list:
            caption = 'Evaluation of the algorithms in the dataset ' + \
                dataset_name + ' with cutoff at ' + str(cutoff) + '. Bold indicates the top-performing algorithm. Underline indicates which algorithm performs best between MF and NNMF, pairwise.\
                long tail cut perc at ' + str(cut_perc)
            label = 'top_n_' + str(cutoff) + '_' + \
                dataset_name + '_' + str(cut_perc)
            first_row = '\\textbf{Algorithm} '
            ml = list(results_list[0][cutoff].keys())
            for metric in ml:
                first_row += '& \\textbf{' + metric.replace('Coverage Test Item', 'CTI').replace(
                    'Diversity - MeanInterList', 'Diversity') + '} '

            if prin:
                print_performance_table(results_list, train, test, validation,
                                        dataset_name, models_names, caption, label, cutoff, first_row, 'c|' + 'c'*len(ml))
            else:
                return results_list


def generate_performance_table():
    """genarate the top-n performance table.
       one can customize the list of cutoffs and the list of metrics
       select the algorithms in the order that is already present in the other
       performance tables, ie: itemknn, userknn, bpr, nnbpr, funk, nnfunk, pmf, nnpmf
    """
    train, test, validation, dataset_name = retrieve_train_validation_test_holdhout_dataset()
    cutoff_list = menu.options([5, 10, 25, 50, 100], [
                               "Cutoff @ 5", "Cutoff @ 10", "Cutoff @ 25", "Cutoff @ 50", "Cutoff @ 100"], title="Pick cutoffs")
    metric_list = menu.options([EvaluatorMetrics.MAP, EvaluatorMetrics.RECALL, EvaluatorMetrics.PRECISION, EvaluatorMetrics.F1, EvaluatorMetrics.NDCG, EvaluatorMetrics.ROC_AUC, EvaluatorMetrics.HIT_RATE, EvaluatorMetrics.ARHR,
                                EvaluatorMetrics.COVERAGE_ITEM_TEST, EvaluatorMetrics.DIVERSITY_MEAN_INTER_LIST],
                               ["MAP", "RECALL", "PRECISION", "F1", "NDCG", "ROC_AUC", "HIT_RATE", "ARHR",
                                "COVERAGE_ITEM_TEST", "DIVERSITY_MEAN_INTER_LIST"],
                               "Pick metrics")
    cuts = menu.options([0.33, 0.66, 1, ], ["Remove long-tail at 0.33", "Remove long-tail at 0.66 (standard long tail cut adopted in the paper)",
                                            "Evaluate on the whole dataset"], title="Pick amount of popular items to take off from the test URM")
    run_on_single_dataset_perf_table(
        train, test, validation, dataset_name, cutoff_list=cutoff_list, metrics_list=metric_list, cut_percs=cuts)


if __name__ == '__main__':
    generate_performance_table()
