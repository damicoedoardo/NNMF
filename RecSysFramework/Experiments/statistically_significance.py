# from RecSysFramework.Experiments.mockup import load_model_mockup
from RecSysFramework.Experiments.significance_test_nnmf import load_significance_split, load_significance_model
from RecSysFramework.Experiments.generate_performance_table import eliminate_item_on_short_head_from_test
from RecSysFramework.Experiments.generate_performance_table import get_result_list
from RecSysFramework.Evaluation.Evaluator import EvaluatorMetrics
from RecSysFramework.Utils import menu
from RecSysFramework.Evaluation.KFold_SignificanceTest import compute_k_fold_significance


def _get_evaluator_metrics_object(metric):
    if metric == 'MAP':
        return EvaluatorMetrics.MAP
    elif metric == 'CTI':
        return EvaluatorMetrics.COVERAGE_ITEM_TEST
    else:
        raise ValueError('Unknown metric.')


def test_statistical_significance(dataset_name, alg, perc, k_splits=10, metric=EvaluatorMetrics.MAP, at=5, save_results=True):
    """test whether the performance difference is statistically significant or not

    Arguments:
        dataset_name {string}
        alg {string} -- tells which two algorithms it should compare. choose among: 'BPR', 'Funk', 'Prob'.
        perc {float} -- tells the percentage of interactions where to cut. choose among: 1, 0.66, 0.33
                        eg: 0.66 remove the top pop items corresponding to the 0.33 of tot interactions, in a classic long tail fashion
                        eg: 1 keeps all the items
        metric {string} -- choose among: EvaluatorMetrics.MAP, EvaluatorMetrics.COVERAGE_ITEM_TEST
        at {int}

    Keyword Arguments:
        k_splits {int} -- (default: {10})
    """
    if alg == 'Funk':
        mf_name = 'FunkSVD'
        nn_name = 'FUNK_NNMF'
    else:
        mf_name = '{}MF'.format(alg)
        nn_name = '{}_NNMF'.format(alg)
    mf_results = []
    nn_results = []
    for k in range(k_splits):
        urm_test = eliminate_item_on_short_head_from_test(load_significance_split(
            dataset_name, k, 'TRAIN'), load_significance_split(dataset_name, k, 'TEST'), perc)
        mf = load_significance_model(dataset_name, k, mf_name)
        nn = load_significance_model(dataset_name, k, nn_name)
        result_list = get_result_list(
            urm_test, [at], [metric], [mf, nn])
        mf_results.append(list(list(result_list[0].values())[0].values())[0])
        nn_results.append(list(list(result_list[1].values())[0].values())[0])

    file = None
    if save_results:
        # open the results file
        file = open('SignificanceTest/results.txt', 'a+')
        file.write('Dataset_name: {}\n'.format(dataset_name))
        file.write('Interactions percentage: {}\n'.format(perc))

    result_dict = compute_k_fold_significance(mf_results, 0.01, nn_results, this_label=mf_name, other_label=nn_name, verbose=True,
                                              log_file=file)


def statistically_significance():
    models = menu.options(['BPR', 'Funk', 'PROB'], ['BPR', 'Funk', 'PROB'],
                          title="Which two models do you want to test against? (MF and NNMF version)")
    datasets = menu.options(['LastFMHetrec2011', 'Movielens1M',
                             'BookCrossing',  'Pinterest', 'CiteULike_a'],
                            ['LastFMHetrec2011', 'Movielens1M', 'BookCrossing',
                             'Pinterest', 'CiteULike_a'], "Pick datasets")
    cutoff_list = menu.options([5, 10, 25, 50, 100], [
        "Cutoff @ 5", "Cutoff @ 10", "Cutoff @ 25", "Cutoff @ 50", "Cutoff @ 100"], title="Pick cutoffs")
    metric_list = menu.options([EvaluatorMetrics.MAP, EvaluatorMetrics.RECALL, EvaluatorMetrics.PRECISION, EvaluatorMetrics.F1, EvaluatorMetrics.NDCG, EvaluatorMetrics.ROC_AUC, EvaluatorMetrics.HIT_RATE, EvaluatorMetrics.ARHR,
                                EvaluatorMetrics.COVERAGE_ITEM_TEST, EvaluatorMetrics.DIVERSITY_MEAN_INTER_LIST],
                               ["MAP", "RECALL", "PRECISION", "F1", "NDCG", "ROC_AUC", "HIT_RATE", "ARHR",
                                "COVERAGE_ITEM_TEST", "DIVERSITY_MEAN_INTER_LIST"],
                               "Pick metrics")
    modes = menu.options([0.33, 0.66, 1, ], ["Remove long-tail at 0.33", "Remove long-tail at 0.66 (standard long tail cut adopted in the paper)",
                                             "Evaluate on the whole dataset"], title="Pick amount of popular items to take off from the test URM")
    for d in datasets:
        for m in modes:
            for c in cutoff_list:
                for me in metric_list:
                    for mo in models:
                        test_statistical_significance(
                            d, mo, m, k_splits=10, metric=me, at=c)


if __name__ == '__main__':
    statistically_significance()
