from RecSysFramework.DataManager.Reader.Movielens1MReader import Movielens1MReader
from RecSysFramework.DataManager.Reader.LastFMHetrec2011Reader import LastFMHetrec2011Reader
from RecSysFramework.DataManager.Reader.CiteULikeReader import CiteULike_aReader
from RecSysFramework.DataManager.Reader.BookCrossingReader import BookCrossingReader
from RecSysFramework.DataManager.Reader.PinterestReader import PinterestReader
import matplotlib.ticker as ticker
import RecSysFramework.Utils.menu as menu
from RecSysFramework.DataManager.Splitter import Holdout
from RecSysFramework.DataManager.Splitter.KFold import KFold
from RecSysFramework.DataManager.DatasetPostprocessing.ImplicitURM import ImplicitURM
from RecSysFramework.DataManager.DatasetPostprocessing.KCore import KCore
from RecSysFramework.Utils import compute_popularity as cp
import pandas as pd
import scipy.sparse as sps
import numpy as np
import matplotlib.pyplot as plt

class DatasetsStatisticsPlotter:
    """
        class used to plot useful statistics of the datasets
    """
    def __init__(self, datasets_analyzers_list):
        # TODO INSERT A CHECK THAT CONFIRMS THAT ALL THE OBJECTS INSIDE THE LIST ARE DATASET ANALYZER
        if not isinstance(datasets_analyzers_list, list):
            datasets_analyzers_list = [datasets_analyzers_list]
        self.datasets_analyzers_list = datasets_analyzers_list

    def get_long_tail_plot(self, interactions_perc=0.33, save_plot=False):
        if not save_plot:
            print('Warning the plot will not be saved!\n to save set the save_plot argument to True')

        #settings final plot
        plt.yscale('log')
        plt.grid(True, which='both', linestyle=':', linewidth=1)
        plt.xlabel('% of interactions', size=15)
        plt.ylabel('% of items', size=15)
        plt.axvline(x=interactions_perc*100, color='black', linestyle='--', linewidth=1.5)
        bbox_props = dict(boxstyle="square,pad=0.3", fc='white', ec='black')
        plt.text(38, 0.005, 'Long-tail\n(unpopular)', bbox=bbox_props, size=13)
        plt.text(8, 0.005, 'Short-head\n(popular)', bbox=bbox_props, size=13)

        long_tails = []

        datsets_names = []
        plotted_lines = []

        for d in self.datasets_analyzers_list:
            long_tails.append(d.get_long_tail_stats(interactions_perc))
            datsets_names.append(d.dataset.get_name())

        for idx, lt in enumerate(long_tails):
            interactions, items = zip(*lt)
            fig = plt.plot(interactions, items, label=datsets_names[idx])
            plotted_lines.append(fig)

        plt.legend(fontsize='medium')

        ax = plt.gca()
        ax.xaxis.set_major_formatter(ticker.PercentFormatter())
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g%%'))

        if save_plot:
            names = '--'.join(datsets_names)
            save_name = 'long_tail_plot_'+names
            plt.savefig(save_name)

        plt.show()


class DatasetAnalyzer:
    """
    class used to retrieve useful statistics of the datasets
    """

    def __init__(self, dataset, postprocessing = None):
        # TODO INSERT A CHECK ON THE OBJECT PASSED IT MUST BE AN OBJECT OF TYPE DATASET
        temp_dataset = dataset

        #Apply postprocessing to the dataset if any
        if postprocessing is not None:
            for postp in postprocessing:
                temp_dataset = postp.apply(dataset=temp_dataset)

        self.dataset = temp_dataset


    def get_statistics(self, latex_output=False):
        """
        return dictionary with basic statistics of the dataset:

        Dataset_name
        Users Interactions: min, max, avg
        Items Interactions: min, max, avg
        density

        if latex_output print the latex table of the statistics
        """
        n_users, n_items = self.dataset.URM_dict["URM_all"].shape

        n_interactions = self.dataset.URM_dict["URM_all"].nnz

        URM_all = sps.csr_matrix(self.dataset.URM_dict["URM_all"])
        user_profile_length = np.ediff1d(URM_all.indptr)

        max_interactions_per_user = user_profile_length.max()
        avg_interactions_per_user = n_interactions / n_users
        min_interactions_per_user = user_profile_length.min()

        URM_all = sps.csc_matrix(self.dataset.URM_dict["URM_all"])
        item_profile_length = np.ediff1d(URM_all.indptr)

        max_interactions_per_item = item_profile_length.max()
        avg_interactions_per_item = n_interactions / n_items
        min_interactions_per_item = item_profile_length.min()

        statistics_dict = {
            'dataset_name': self.dataset.get_name(),
            'n_user': n_users,
            'n_items': n_items,
            'n_interactions': n_interactions,
            'min_interactions_per_user': min_interactions_per_user,
            'max_interactions_per_user': max_interactions_per_user,
            'avg_interactions_per_user': round(avg_interactions_per_user, 2),
            'min_interactions_per_item': min_interactions_per_item,
            'max_interactions_per_item': max_interactions_per_item,
            'avg_interactions_per_item': round(avg_interactions_per_item, 2)
        }

        if latex_output:
            print(
                " \\begin{{table}} \n \
            \\centering \n \
                    \\caption{{dataset {} statistics}} \n \
                    \\label{{table:dataset {} statistics}} \n \
                    \\resizebox{{1\\linewidth}}{{!}}{{% \n \
                    \\begin{{tabular}}{{c|ccc|ccc|ccc|c}} \n \
                    \\toprule \n \
                    \\textbf{{Dataset}} & Users & Items & Interactions & \multicolumn{{3}}{{c|}}{{Users Interactions}}&\\multicolumn{{3}}{{c|}}{{Items Interactions}}& density\\\\ \n \
                                                                                                                                                                                    &&&&min&max&avg&min&max&avg&&\\\\[-3ex] \n \
                    \\midrule \n \
                    {} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {}\%\\\\ \n \
                    \\bottomrule \n \
                    \\end{{tabular}} \n \
                    }} \n \
                    \\end{{table}} \n \
                    ".format(self.dataset.get_name(),
                             self.dataset.get_name(),
                             self.dataset.get_name(),
                             n_users,
                             n_items,
                             n_interactions,
                             min_interactions_per_user,
                             max_interactions_per_user,
                             round(avg_interactions_per_user, 2),
                             min_interactions_per_item,
                             max_interactions_per_item,
                             round(avg_interactions_per_item, 2),
                             round(n_interactions / (n_items * n_users) * 100, 2),
                             )
            )
            return statistics_dict

    def get_long_tail_stats(self, interactions_perc=0.33):
        """
        :param interactions_perc: long tail cut considered
        default is 0.33 meaning it will consider in the long tail the items accounting for the 66 perc of interactions
        and in the popular short head the items accounting for the 33 perc of total interactions

        return [(0.1,0.01)...(%interactions, %items)...(100,100)]
        """
        urm = self.dataset.get_URM()

        #computing the item popularity
        item_pop_tuple_list = cp.compute_popularity_item(urm)[::-1]
        items_idxs, interactions = zip(*item_pop_tuple_list)

        #compute the cumulative function over the interactions
        interactions_cumsum = np.cumsum(interactions)
        interactions_cumsum_norm = interactions_cumsum / max(interactions_cumsum)

        #compute the number of items that accounts for the percentage of interactions in a cell of interactions_cum_sum_norm
        #and notmalize them
        items_number = np.cumsum(np.ones(len(interactions_cumsum_norm)))
        items_number_norm = items_number/len(items_idxs)

        items_interactions_percentage = list(zip(interactions_cumsum_norm*100, items_number_norm*100))
        return items_interactions_percentage

if __name__ == '__main__':
    procede = True
    an_list = []
    while procede:
        dataset = menu.single_choice('select the Dataset', ['Movielens1MReader', 'LastFMHetrec2011Reader',
                                                            'CiteULike_aReader', 'BookCrossingReader', 'PinterestReader'])
        reader = eval(dataset)()
        ds = reader._load_from_original_file()
        implicit_param = int(input('Implicit\n'))
        implicit = ImplicitURM(implicit_param)
        kcore = KCore(item_k_core=5, user_k_core=5)
        an_list.append(DatasetAnalyzer(ds, postprocessing=[implicit, kcore]))

        if input('Want another dataset?\n') == 'n':
            procede = False

    mode = menu.single_choice('What do you want to do?', labels=['long tail', 'base stats'])
    if mode == 'base stats':
        for a in an_list:
            a.get_statistics(latex_output=True)
    elif mode == 'long tail':
        DatasetsStatisticsPlotter(an_list).get_long_tail_plot(save_plot=True)
