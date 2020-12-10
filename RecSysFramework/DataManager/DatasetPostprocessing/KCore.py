#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 30/11/18

@author: XXX
"""


from .DatasetPostprocessing import DatasetPostprocessing


class KCore(DatasetPostprocessing):

    """
    This class selects a dense partition of URM such that all items and users have at least K interactions.
    The algorithm is recursive and might not converge until the graph is empty.
    https://www.geeksforgeeks.org/find-k-cores-graph/
    """

    def __init__(self, user_k_core, item_k_core, reshape=True):

        assert user_k_core >= 1,\
            "DatasetPostprocessingKCore: user_k_core must be a positive value >= 1, provided value was {}".format(user_k_core)

        assert item_k_core >= 1,\
            "DatasetPostprocessingKCore: item_k_core must be a positive value >= 1, provided value was {}".format(item_k_core)

        super(KCore, self).__init__()
        self.user_k_core = user_k_core
        self.item_k_core = item_k_core
        self.reshape = reshape


    def get_name(self):
        return "kcore_user_{}_item_{}{}".format(self.user_k_core, self.item_k_core, "_reshaped" if self.reshape else "")


    def apply(self, dataset):

        from RecSysFramework.DataManager.Utils import select_asymmetric_k_cores

        _, removedUsers, removedItems = select_asymmetric_k_cores(dataset.get_URM(), user_k_value=self.user_k_core,
                                                                  item_k_value=self.item_k_core, reshape=False)

        new_dataset = dataset.copy()
        new_dataset.remove_items(removedItems, keep_original_shape=not self.reshape)
        new_dataset.remove_users(removedUsers, keep_original_shape=not self.reshape)
        new_dataset.add_postprocessing(self)

        return new_dataset
