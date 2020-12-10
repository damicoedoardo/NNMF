# from RecSysFramework.Experiments.mockup import load_model_mockup
from RecSysFramework.Experiments.significance_test_nnmf import load_significance_model
import numpy as np
from RecSysFramework.Experiments.utils_experiments import round_all_digits


def return_training_time_table(dataset_name, k_splits=10):
    models = ['BPRMF', 'BPR_NNMF', 'FunkSVD',
              'FUNK_NNMF', 'PROBMF', 'PROB_NNMF']
    times = []
    stds = []
    for model in models:
        time_model = np.zeros(k_splits)
        for k in range(k_splits):
            # qui mettere il metodo di edo ;)

            m = load_significance_model(dataset_name, k, model)
            # m = load_model_mockup(dataset_name, k, model)
            time_model[k] = m._elapsed_time_training
        times.append(np.mean(time_model)/60)
        stds.append(np.std(time_model)/60)
    return models, times, stds


def print_training_time_table():
    datasets = ['LastFMHetrec2011', 'Movielens1M',
                'BookCrossing', 'Pinterest', 'CiteULike-a']
    datasets_codes = ['\\lastfm', '\\movielensom',
                      '\\bookcrossing', '\\pinterest', '\\citulike']
    heading = "\\textbf{Algorithm}"
    d = {
        'BPRMF': "\\bprmf",
        'BPR_NNMF': "\\bprnnmf",
        'FunkSVD': "\\funkmf",
        'FUNK_NNMF': "\\funknnmf",
        'PROBMF': "\\pmf",
        'PROB_NNMF': "\\pnnmf"
    }
    for ds, dc in zip(datasets, datasets_codes):
        heading += " & \\textbf{" + dc + "}"
        models, means, stds = return_training_time_table(ds)
        for m, n, s in zip(models, means, stds):
            d[m] += f" & {round_all_digits(n, 1)} ± {round_all_digits(s, 3)}"

    table = "\\begin{table*}[h] \\centering \\caption{Training time in minutes.} \\begin{tabular}{c" + \
        "|c"*len(datasets) + "} \\toprule "
    table += f"{heading} \\\\ \\midrule \n"
    table += f"{d['BPRMF']} \\\\ \n"
    table += f"{d['BPR_NNMF']} \\\\ \\midrule \n"
    table += f"{d['FunkSVD']} \\\\ \n"
    table += f"{d['FUNK_NNMF']} \\\\ \\midrule \n"
    table += f"{d['PROBMF']} \\\\ \n"
    table += f"{d['PROB_NNMF']} \\\\ \\bottomrule \n"

    table += "\\end{tabular} \\label{table:training_time} \end{table*}"

    print(table)


if __name__ == '__main__':
    print_training_time_table()
