
from RecSysFramework.Recommender.MatrixFactorization.MatrixFactorization_Cython import BPR_NNMF, FUNK_NNMF, PROB_NNMF
from RecSysFramework.Evaluation.Evaluator import EvaluatorHoldout
import numpy as np
from RecSysFramework.Utils.get_holdout import retrieve_train_validation_test_holdhout_dataset
import matplotlib.pyplot as plt
from RecSysFramework.Utils.check_folder import check_folder
from RecSysFramework.Utils import menu
from RecSysFramework.Experiments.difference_of_recs_among_same_models import get_params_dict_model_dataset
from collections import defaultdict


def smooth(y):
    w = 0.6
    y_filtered = []
    for idx, v in enumerate(y):
        if idx == 0:
            y_filtered.append(v)
        else:
            y_filtered.append(w*y_filtered[idx-1]+(1-w)*v)
    return y_filtered


def plot():
    exp_kind = 'Optimal parameters'
    models_names = menu.options(['BPRMF', 'FunkSVD', 'probmf'], ['BPR', 'Funk', 'Prob'])
    dataset_name_raw = menu.single_choice('Select the dataset you want to create',
                                          ['Movielens1MReader', 'LastFMHetrec2011Reader', 'BookCrossingReader', 'CiteULike_aReader',
                                           'PinterestReader'])
    dataset_name = dataset_name_raw.replace('Reader', '').replace('CiteULike_a', 'CiteULike-a')

    for model_name in models_names:
        for plot_kind in ['MAP on validation', 'Loss on train']:

            try:
                d1 = np.load(f"DataExperiments/Convergence/{dataset_name}/{exp_kind}/{model_name}_MF.npy", allow_pickle=True).item()
                d2 = np.load(f"DataExperiments/Convergence/{dataset_name}/{exp_kind}/{model_name}_NNMF.npy", allow_pickle=True).item()
            except FileNotFoundError:
                save(exp_kind, models_names, dataset_name_raw)

            if plot_kind == 'MAP on validation':
                key_1 = 'metric_validation'
                key_2 = 'MAP'
                x_1 = [0]
                x_2 = [0]
                y_1 = [0]
                y_2 = [0]
            else:
                key_1 = 'metric_train'
                key_2 = 'train_loss'
                x_1 = []
                x_2 = []
                y_1 = []
                y_2 = []

            epochs_counter = 5
            while True:
                try:
                    val_y_1 = d1[key_1][epochs_counter][key_2]
                except KeyError:
                    break
                x_1.append(epochs_counter)
                y_1.append(val_y_1)
                epochs_counter += 5

            epochs_counter = 5
            while True:
                try:
                    val_y_2 = d2[key_1][epochs_counter][key_2]
                except KeyError:
                    break
                x_2.append(epochs_counter)
                y_2.append(val_y_2)
                epochs_counter += 5

            font = {'family': 'normal',
                    'size': 15}

            plt.rc('font', **font)

            fig = plt.figure()
            fig.tight_layout()

            plt.plot(x_1, smooth(y_1), "-", color='dimgray', label="MF", linewidth=3)
            plt.plot(x_2, smooth(y_2), "--", color='dimgray', label="NNMF", linewidth=3)
            plt.legend()
            plt.grid()
            if model_name == 'BPRMF':
                plt.title("BPR")
            elif model_name == 'FunkSVD':
                plt.title("Funk")
            elif model_name == 'probmf':
                plt.title("Probabilistic")
            plt.xlabel("Epochs")
            y_label = "MAP" if plot_kind == 'MAP on validation' else 'Loss'
            plt.ylabel(y_label)

            exp_kind_name = 'opt' if exp_kind == 'Optimal parameters' else 'non-opt'
            plot_kind_name = 'MAP' if plot_kind == 'MAP on validation' else 'Loss'
            plt.savefig(f"{dataset_name}_{exp_kind_name}_{plot_kind_name}_{model_name}.png", bbox_inches="tight")
            print(f"Figure saved in {dataset_name}_{exp_kind_name}_{plot_kind_name}_{model_name}.png")


def save(exp_kind, models_names, dataset_name):

    def _save_dictionary_model(algorithm, dataset_name, exp_kind, model_name, mf_or_nnmf):
        # d = { metric_validation : {
        #            5 : {
        #                            metric1: val,
        #                            .. ,
        #                            metricn: val },
        #            10 : {
        #                            metric1: val,
        #                            .. ,
        #                            metricn: val },
        path = f"DataExperiments/Convergence/{dataset_name}/{exp_kind}/"
        check_folder(path)

        d = {'metric_validation': algorithm._epoch_result_dictionary}

        loss_history = algorithm.cythonEpoch.get_loss_history()
        d["metric_train"] = defaultdict()
        for i in range(len(loss_history)):
            d["metric_train"][i] = {"train_loss": loss_history[i]}

        np.save(f"{path}/{model_name}_{mf_or_nnmf}", d)

    train, test, validation, dataset_name = retrieve_train_validation_test_holdhout_dataset(dataset_name=dataset_name)

    evalh = EvaluatorHoldout([5])
    evalh.global_setup(URM_test=validation.get_URM())

    for model_name in models_names:
        if model_name == 'BPRMF':
            algorithm = BPR_NNMF(train.get_URM())
        elif model_name == 'FunkSVD':
            algorithm = FUNK_NNMF(train.get_URM())
        elif model_name == 'probmf':
            algorithm = PROB_NNMF(train.get_URM())

        params_dict = get_params_dict_model_dataset(
            dataset_name, model_name, evalh)
        params_dict['validation_every_n'] = 5
        params_dict["verbose"] = True
        params_dict["stop_on_validation"] = False

        algorithm.fit(**params_dict)
        _save_dictionary_model(algorithm, dataset_name, exp_kind, model_name, 'MF')

       # Clone model
        if model_name == 'BPRMF':
            algorithm = BPR_NNMF(train.get_URM())
        elif model_name == 'FunkSVD':
            algorithm = FUNK_NNMF(train.get_URM())
        elif model_name == 'probmf':
            algorithm = PROB_NNMF(train.get_URM())

        if exp_kind == 'Comparable number of updates':
            params_dict['item_k'] = 5
            params_dict['user_k'] = 5
        elif exp_kind == 'Optimal parameters':
            nn_version_name = model_name.replace('probmf', 'nnprobmf').replace(
                'FunkSVD', 'FUNKMF').replace('MF', '_NNMF')
            params_dict = get_params_dict_model_dataset(
                dataset_name, nn_version_name, evalh)
        params_dict['validation_every_n'] = 5
        params_dict["verbose"] = True
        params_dict["stop_on_validation"] = False

        algorithm.fit(**params_dict)
        _save_dictionary_model(algorithm, dataset_name, exp_kind, model_name, 'NNMF')


if __name__ == '__main__':
    plot()
