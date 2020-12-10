from RecSysFramework.DataManager.DatasetPostprocessing import DatasetPostprocessing
from RecSysFramework.Utils import compute_popularity as cp
import numpy as np


class LongQueueAnalysis(DatasetPostprocessing):
    """
    remove from the dataset the interaction with the most popular items
    item_perc is the percentage of less popular items to mantain in the dataset
    """

    def __init__(self, item_perc=0.66, reshape=True):

        assert 0 < item_perc < 1, 'percentage of item to keep has to be between 0<item_perc<1'

        self.item_perc=0.66
        self.reshape = reshape

    def get_name(self):
        return "Long Queue Analysis percentage item mantained:{}{}".format(self.item_perc,
                                                                            '_reshaped' if self.reshape else '')

    def apply(self, dataset):
        def get_items_to_remove(urm, item_perc):
            """
            return the list of items idxs to remove
            """
            items_idxs, interactions = zip(*cp.compute_popularity_item(urm))
            cumsum_interactions = np.cumsum(interactions)
            cumsum_interactions_norm = cumsum_interactions / max(cumsum_interactions)
            cut_idx = (np.abs(cumsum_interactions_norm - item_perc)).argmin()
            items_to_remove = list(items_idxs[cut_idx+1:])

            print('Items Before preprocessing: {}'.format(len(items_idxs)))
            print('Items After preprocessing: {}'.format(cut_idx))
            print('Remaining items percentage:{}\n'.format((cut_idx)/len(items_idxs)))

            print('Interactions Before Preprocessing: {}'.format(cumsum_interactions[-1]))
            print('Interactions After Preprocessing: {}'.format(cumsum_interactions[cut_idx]))
            print('Remaining interactions Percentage: {}'.format((cumsum_interactions[cut_idx])/
                                                                 cumsum_interactions[-1]))

            return items_to_remove

        items_to_remove = get_items_to_remove(dataset.get_URM(), self.item_perc)
        new_dataset = dataset.copy()
        new_dataset.remove_items(items_to_remove, keep_original_shape=not self.reshape)
        new_dataset.add_postprocessing(self)

        return new_dataset


