from RecSysFramework.Experiments.difference_of_recs_among_same_models import compute_recommendations_similarity, get_params_dict_model_dataset
from RecSysFramework.Recommender.MatrixFactorization.MatrixFactorization_Cython import BPR_NNMF, FUNK_NNMF, PROB_NNMF
from RecSysFramework.Evaluation.Evaluator import EvaluatorHoldout
import numpy as np
from RecSysFramework.Utils.get_holdout import retrieve_train_validation_test_holdhout_dataset
import math
from matplotlib.ticker import StrMethodFormatter
import matplotlib.pyplot as plt


def save_stability_results(dataset_name):
    exp_kind = 'Optimal parameters'
    choice = 'representations'

    train, test, validation, dataset_name = retrieve_train_validation_test_holdhout_dataset(dataset_name=dataset_name)

    # Evaluator is not really used in this test
    evalh = EvaluatorHoldout([5])
    evalh.global_setup(URM_test=test.get_URM())

    results_list_recommendation = []
    results_list_representations = []
    model_names = []

    cuts_short_head = [1, 0.66, 0.4, 0.3, 0.2, 0.1, 0.0]
    for model_name in ['BPRMF', 'FunkSVD', 'probmf']:
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
            cutoffs_jaccard=[1, 5, 10, 25], cuts=cuts_short_head, choice=choice, exp_kind=exp_kind, eval_on_pop=False, eval_on_range=True, cutoffs_representations=[100])

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
            cutoffs_jaccard=[1, 5, 10, 25], cuts=cuts_short_head, choice=choice, exp_kind=exp_kind, eval_on_pop=False, eval_on_range=True, cutoffs_representations=[100])

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

    path = f"DataExperiments/StabilityShortHead/{dataset_name}"
    np.save(path, results_list_representations)


def plot_stability_short_head():
    train, test, validation, dataset_name = retrieve_train_validation_test_holdhout_dataset()
    try:
        stab_values = np.load(
            f"DataExperiments/StabilityShortHead/{dataset_name}.npy", allow_pickle=True)
    except FileNotFoundError:
        save_stability_results(dataset_name)

    """
    Stabiity values, each elem is the results for an algorithm
    0) BPR
    1) NNBPR
    2) Funk
    3) NNFunk
    4) Prob
    5) NNProb
    The stability values are stored in a dictionary with this format:
        jaccard_{thrhigh}-{thrlow}_10_Jaccard (users|items)
    """

    thrs = [1, 0.66, 0.4, 0.3, 0.2, 0.1, 0.0]
    stab_models = []
    for v in [0, 1, 2, 3, 4, 5]:
        d = stab_values[v]
        values = []
        for thr in zip(thrs, thrs[1:]):
            values.append(d[0][f"jaccard_{thr[0]}-{thr[1]}_100_Jaccard items"])
        stab_models.append(values)

    fig = plt.figure()

    font = {'family': 'normal',
            'size': 21}

    plt.rc('font', **font)

    plt.xlabel('Item popularity bin')
    plt.ylabel('Stability')

    plt.plot([r+0.5 for r in range(len(stab_models[0]))],
             stab_models[0], "D--", color='dimgray', label="MF", linewidth=3, markersize=10)
    plt.plot([r+0.5 for r in range(len(stab_models[0]))],
             stab_models[1], "o-", color='dimgray', label="NNMF", linewidth=3, markersize=10)
    plt.xticks(range(len(stab_models[0])+1), labels=['1', '0.66', '0.4', '0.3', '0.2', '0.1', '0'])
    plt.yticks(np.arange(min(min(stab_models[0], stab_models[1])), max(max(stab_models[2], stab_models[3])), 0.05))
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    plt.legend()
    plt.grid()
    plt.savefig(
        f"{dataset_name.replace('Reader','')}_stability_varying_popularity_BPR.png", bbox_inches="tight")
    print(f"Figure saved as {dataset_name.replace('Reader','')}_stability_varying_popularity_BPR.png")

    fig = plt.figure()

    font = {'family': 'normal',
            'size': 21}

    plt.rc('font', **font)

    plt.xlabel('Item popularity bin')
    plt.ylabel('Stability')

    plt.plot([r+0.5 for r in range(len(stab_models[0]))],
             stab_models[2], "D--", color='dimgray', label="MF", linewidth=3, markersize=10)
    plt.plot([r+0.5 for r in range(len(stab_models[0]))],
             stab_models[3], "o-", color='dimgray', label="NNMF", linewidth=3, markersize=10)
    plt.yticks(np.arange(min(min(stab_models[2], stab_models[3])), max(max(stab_models[2], stab_models[3])), 0.05))
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    plt.xticks(range(len(stab_models[0])+1), labels=['1', '0.66', '0.4', '0.3', '0.2', '0.1', '0'])
    plt.legend()
    plt.grid()
    plt.savefig(
        f"{dataset_name.replace('Reader','')}_stability_varying_popularity_Funk.png", bbox_inches="tight")
    print(f"Figure saved as {dataset_name.replace('Reader','')}_stability_varying_popularity_Funk.png")

    fig = plt.figure()

    font = {'family': 'normal',
            'size': 21}

    plt.rc('font', **font)

    plt.xlabel('Item popularity bin')
    plt.ylabel('Stability')

    plt.plot([r+0.5 for r in range(len(stab_models[0]))],
             stab_models[4], "D--", color='dimgray', label="MF", linewidth=3, markersize=10)
    plt.plot([r+0.5 for r in range(len(stab_models[0]))],
             stab_models[5], "o-", color='dimgray', label="NNMF", linewidth=3, markersize=10)
    plt.xticks(range(len(stab_models[0])+1), labels=['1', '0.66', '0.4', '0.3', '0.2', '0.1', '0'])
    plt.yticks(np.arange(min(min(stab_models[4], stab_models[5])), max(max(stab_models[2], stab_models[3])), 0.05))
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    plt.legend()
    plt.grid()
    plt.savefig(
        f"{dataset_name.replace('Reader','')}_stability_varying_popularity_P.png", bbox_inches="tight")
    print(f"Figure saved as {dataset_name.replace('Reader','')}_stability_varying_popularity_P.png")


if __name__ == "__main__":
    plot_stability_short_head()
