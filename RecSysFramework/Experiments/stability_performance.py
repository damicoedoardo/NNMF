from RecSysFramework.Evaluation.Evaluator import EvaluatorMetrics, EvaluatorUsersByPopularity
from RecSysFramework.Utils.get_holdout import retrieve_train_validation_test_holdhout_dataset
from RecSysFramework.Recommender.MatrixFactorization.MatrixFactorization_Cython import BPR_NNMF, FUNK_NNMF, PROB_NNMF
from RecSysFramework.Experiments.generate_performance_table import eliminate_item_on_short_head_from_test
from RecSysFramework.Utils.check_folder import check_folder
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from RecSysFramework.Utils import menu
from collections import defaultdict


def save():
    train, test, validation, dataset_name = retrieve_train_validation_test_holdhout_dataset()
    models_names = ['BPRMF', 'FunkSVD', 'probmf']

    cutoff_list = [5, 10, 20]
    metric_list = [EvaluatorMetrics.MAP, EvaluatorMetrics.RECALL]

    thrs = [1, 0.66, 0.4, 0.3, 0.2, 0.1, 0.0]
    evalh = EvaluatorUsersByPopularity(cutoff_list, metric_list, thr=thrs)
    evalh.global_setup(test.get_URM())

    d = defaultdict(list)

    for model_name in models_names:

        if model_name == 'BPRMF':
            key = 'BPR'
            algorithm = BPR_NNMF(train.get_URM())
        elif model_name == 'FunkSVD':
            key = 'Funk'
            algorithm = FUNK_NNMF(train.get_URM())
        elif model_name == 'probmf':
            key = 'P'
            algorithm = PROB_NNMF(train.get_URM())

        for alg in ['MF', 'NNMF']:

            model_path = 'StabilityExperimentsModels/{}/{}/'.format(dataset_name, algorithm.RECOMMENDER_NAME)

            for idx in range(10):

                if alg == 'NNMF':
                    file_name = '{}_{}_opt'.format(alg, idx)
                else:
                    file_name = '{}_{}'.format(alg, idx)

                algorithm.load_model(folder_path=model_path, file_name=file_name)

                results_list = evalh.evaluateRecommender(algorithm)

                for t in zip(thrs, thrs[1:]):
                    d[f"{key}-{alg}-({t[0]}, {t[1]})"].append(results_list[t].get_results_dictionary())

        for metric in metric_list:
            for cutoff in cutoff_list:
                metric_name = metric.name.replace('RECALL', 'Recall')

                to_save = {}

                for key in d.keys():

                    to_save[key] = []

                    for r in d[key]:
                        to_save[key].append(r[cutoff][metric_name])

                check_folder(f"DataExperiments/StabilityPerformance/{dataset_name}/")
                np.save(f"DataExperiments/StabilityPerformance/{dataset_name}/{metric_name}@{cutoff}", to_save)


def load_dict(dataset, metric, cutoff):
    """
    load the dictionary associated to:
    * dataset
    * metric
    * cutoff
    """
    base_path = "DataExperiments/StabilityPerformance/"
    dict_path = base_path + "{}/{}@{}.npy".format(dataset, metric, cutoff)
    return np.load(dict_path, allow_pickle=True)


def performance_box_plot(dataset, metric, cutoff, save=True):
    """
    Create the box-plot distribution of a metric at a given cutoff for a given dataset

    The multiple values has been taken changing the random initialization seed
    """
    plt.figure()

    data_dict = load_dict(dataset, metric, cutoff).item()

    data_reworked = []
    for k, value_list in data_dict.items():
        alg, kind, upop = k.split("-")
        for v in value_list:
            data_reworked.append((v, alg, kind, upop))
    metric_cutoff = "{}@{}".format(metric, cutoff)
    data_df = pd.DataFrame(data_reworked, columns=[metric_cutoff, "Algorithm", "kind", "User popularity bin"])

    for model in ["BPR", "Funk", "P"]:

        sns.set(style="whitegrid")
        ax = sns.boxplot(x="User popularity bin", y=metric_cutoff, hue="kind", data=data_df[data_df.Algorithm == model], linewidth=0.8)
        ax.set(xlabel="")
        ax.set_title(model)
        ax.legend().set_title("")
        plt.show()

        # longtail_cut = str(longtail_cut).replace('.', '')

        # if save:
        #     plt.savefig("DataExperiments/StabilityPerformance/{}_boxplot_distr_metric_{}@{}_ltc{}".format(dataset, metric, cutoff, longtail_cut))


if __name__ == "__main__":
    # save()
    datasets_list = [
        "BookCrossing",
        "CiteULike-a",
        "LastFMHetrec2011",
        "Movielens1M",
        "Pinterest",
    ]
    metric_list = ["MAP", "Recall"]
    cutoff_list = ["5", "20"]
    dataset = menu.single_choice("Select Dataset", datasets_list)

    for metric in metric_list:
        for cutoff in cutoff_list:
            performance_box_plot(dataset, metric, cutoff, save=True)
