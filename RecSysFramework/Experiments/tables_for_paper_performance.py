from RecSysFramework.Experiments.generate_performance_table import run_on_single_dataset_perf_table
from RecSysFramework.Utils.get_holdout import retrieve_train_validation_test_holdhout_dataset
from RecSysFramework.tables_for_paper_stability import round_all_digits
from RecSysFramework.Evaluation.Evaluator import EvaluatorMetrics


def tables_for_paper_performance():
    t = "\\begin{tabular}{c | ccccccc}\
         \\toprule\
         \\textbf{Algorithm} & \\textbf{LastFM} &\\textbf{Mov1M} &\\textbf{BCrossing} &\
         \\textbf{Pinterest} &\\textbf{CiteUL} \\midrule "

    r = []

    l = ['LastFMHetrec2011Reader', 'Movielens1MReader',
         'BookCrossingReader', 'PinterestReader', 'CiteULike_aReader', ]
    algs = ["TopPop", "ItemKNNCF", "UserKNNCF", "SLIM BPR",
            "PureSVD", "BPRMF", "BPR NNMF", "FunkSVD", "Funk NNMF", "ProbMF", "Prob NNMF"]

    # l = ["LastFMHetrec2011Reader"]
    # algs = ["TopPop", "ItemKNNCF"]

    for d in l:
        train, test, validation, dataset_name = retrieve_train_validation_test_holdhout_dataset(
            dataset_name=d)
        cutoff_list = [5, 20]
        metric_list = [EvaluatorMetrics.MAP, EvaluatorMetrics.RECALL]
        cuts = [0.66]
        r.append(run_on_single_dataset_perf_table(
            train, test, validation, dataset_name, cutoff_list=cutoff_list, metrics_list=metric_list, cut_percs=cuts, prin=False))

    t = ""
    for idx, alg in enumerate(algs):
        if idx == 0:
            continue
        else:
            line = alg
            for res in r:
                alg_res = res[idx]
                for metric in metric_list:
                    metric = metric.value.METRIC_NAME
                    for cutoff in cutoff_list:
                        line += f" & {round_all_digits(alg_res[cutoff][metric], 3)}"
            t += f"{line} \\\\ \\midrule \n"
    t += "\\bottomrule \\end{tabular}"
    print(t)


if __name__ == "__main__":
    tables_for_paper_performance()
