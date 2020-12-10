#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 30/11/18

@author: XXX
"""

import numpy as np
import scipy.sparse as sps
import os

from RecSysFramework.DataManager import Dataset
from RecSysFramework.DataManager.Splitter.Holdout import Holdout

from .DataSplitter import DataSplitter


class KFold(DataSplitter):
    """
    The splitter tries to load from the specific folder related to a dataset, a split in the format corresponding to
    the splitter class. Basically each split is in a different subfolder
    - The "original" subfolder contains the whole dataset, is composed by a single URM with all data and may contain
        ICMs as well, either one or many, depending on the dataset
    - The other subfolders "warm", "cold" ecc contains the splitted data.

    The dataReader class involvement is limited to the following cased:
    - At first the dataSplitter tries to load from the subfolder corresponding to that split. Say "warm"
    - If the dataReader is succesful in loading the files, then a split already exists and the loading is complete
    - If the dataReader raises a FileNotFoundException, then no split is available.
    - The dataSplitter then creates a new instance of dataReader using default parameters, so that the original data will be loaded
    - At this point the chosen dataSplitter takes the URM_all and selected ICM to perform the split
    - The dataSplitter saves the splitted data in the appropriate subfolder.
    - Finally, the dataReader is instantiated again with the correct parameters, to load the data just saved
    """

    def __init__(self, n_folds=5, forbid_new_split=False, force_new_split=False, allow_cold_users=False,
                 test_rating_threshold=0, random_seed=42, percentage_initial_data_to_split=0.8):
        """

        :param dataReader_object:
        :param n_folds:
        :param force_new_split:
        :param forbid_new_split:
        :param save_folder_path:    path in which to save the loaded dataset
                                    None    use default "dataset_name/split_name/"
                                    False   do not save
        """

        assert n_folds > 1, "DataSplitterKFold: Number of folds must be  greater than 1"

        super(KFold, self).__init__(forbid_new_split=forbid_new_split, force_new_split=force_new_split,
                                    allow_cold_users=allow_cold_users, with_validation=False, random_seed=random_seed)
        self.sequential_split_number = 0
        self.test_rating_threshold = test_rating_threshold
        self.n_folds = n_folds
        self.percentage_initial_data_to_split = percentage_initial_data_to_split

    def reset_split_number(self):
        self.sequential_split_number = 0


    def load_split(self, datareader, save_folder_path=None, postprocessings=None):

        tmp_save_folder_path = save_folder_path
        if tmp_save_folder_path is None:
            tmp_save_folder_path = datareader.get_complete_default_save_path(postprocessings)

        try:

            datalist = self._get_dataset_names_in_split()
            for i in range(self.n_folds):
                for d in datalist:
                    if not datareader.all_files_available(tmp_save_folder_path + self.get_name() + os.sep,
                                                          filename_suffix="_{}_{}".format(i, d)):
                        raise Exception
            r = []
            for i in range(self.n_folds):

                datasets = []
                for d in datalist:
                    urm, urm_mappers, icm, icm_mappers, ucm, ucm_mappers = datareader.load_from_saved_sparse_matrix(
                        tmp_save_folder_path + self.get_name() + os.sep, filename_suffix="_{}_{}".format(i, d))
                    datasets.append(Dataset(datareader.get_dataset_name(),
                                            base_folder=datareader.get_default_save_path(),
                                            postprocessings=postprocessings,
                                            URM_dict=urm, URM_mappers_dict=urm_mappers,
                                            ICM_dict=icm, ICM_mappers_dict=icm_mappers,
                                            UCM_dict=ucm, UCM_mappers_dict=ucm_mappers))

                # With KFold, Validation is intrisic, so we surely have only train and test datasets
                r.append((datasets[0], datasets[1]))
            
            return r
        except:

            print("DataSplitterKFold: Preloaded data not found or corrupted, reading from original files...")
            dataset = datareader.load_data(save_folder_path=save_folder_path, postprocessings=postprocessings)
            return self.split(dataset)


    def save_split(self, split, save_folder_path=None, filename_suffix=""):
        for split_number in range(len(split)):
            super(KFold, self).save_split(split[0], save_folder_path,
                                        filename_suffix="{}_{}".format(filename_suffix, split_number))


class WarmItemsKFold(KFold):
    """
    The splitter tries to load from the specific folder related to a dataset, a split in the format corresponding to
    the splitter class. Basically each split is in a different subfolder
    - The "original" subfolder contains the whole dataset, is composed by a single URM with all data and may contain
        ICMs as well, either one or many, depending on the dataset
    - The other subfolders "warm", "cold" ecc contains the splitted data.

    The dataReader class involvement is limited to the following cased:
    - At first the dataSplitter tries to load from the subfolder corresponding to that split. Say "warm"
    - If the dataReader is succesful in loading the files, then a split already exists and the loading is complete
    - If the dataReader raises a FileNotFoundException, then no split is available.
    - The dataSplitter then creates a new instance of dataReader using default parameters, so that the original data will be loaded
    - At this point the chosen dataSplitter takes the URM_all and selected ICM to perform the split
    - The dataSplitter saves the splitted data in the appropriate subfolder.
    - Finally, the dataReader is instantiated again with the correct parameters, to load the data just saved
    """

    def get_name(self):
        return "warm_items_{}_fold_testthreshold_{:.1f}{}".format(self.n_folds, self.test_rating_threshold,
                                                                  "" if self.allow_cold_users else "_no_cold_users")

    def split(self, dataset):

        super(WarmItemsKFold, self).split(dataset)

        # I can do the kfold of a slice of the initial URM!
        if self.percentage_initial_data_to_split < 1.0:
            h = Holdout(train_perc=self.percentage_initial_data_to_split, test_perc=1-self.percentage_initial_data_to_split)
            dataset = h.split(dataset)[0]

        folds = []
        URM = dataset.get_URM().tocoo()
        split_belonging = np.random.choice(self.n_folds, URM.data.size, replace=True)

        for i in range(self.n_folds):

            urm = {}
            urm_mappers = {}
            mask = split_belonging == i
            for URM_name in dataset.get_URM_names():
                URM = dataset.get_URM(URM_name).tocoo()
                # Sort nnz values by row and column indices, in order to remain consistent in the splits of different URMs
                row, col, data = zip(*sorted(zip(URM.row, URM.col, URM.data), key=lambda x: (x[0], x[1])))
                urm[URM_name] = sps.csr_matrix((np.array(data)[mask], (np.array(row)[mask], np.array(col)[mask])),
                                               shape=URM.shape)
                urm_mappers[URM_name] = dataset.get_URM_mapper(URM_name)

            folds.append(
                Dataset(dataset.get_name(), base_folder=dataset.get_base_folder(),
                        postprocessings=dataset.get_postprocessings(),
                        URM_dict=urm, URM_mappers_dict=urm_mappers,
                        ICM_dict=dataset.get_ICM_dict(), ICM_mappers_dict=dataset.get_ICM_mappers_dict(),
                        UCM_dict=dataset.get_UCM_dict(), UCM_mappers_dict=dataset.get_UCM_mappers_dict()
                        )
            )

        r = []
        for i in range(self.n_folds):
            urm = {}
            urm_mappers = {}
            for URM_name in folds[i].get_URM_names():
                # Keep i-th fold as test and merge the others as train
                urm[URM_name] = folds[(i + 1) % self.n_folds].get_URM(URM_name)
                urm_mappers[URM_name] = folds[(i + 1) % self.n_folds].get_URM_mapper(URM_name)
                for j in range(2, self.n_folds):
                    urm[URM_name] += folds[(i + j) % self.n_folds].get_URM(URM_name)

            train = Dataset(folds[i].get_name(), base_folder=folds[i].get_base_folder(),
                            postprocessings=folds[i].get_postprocessings(),
                            URM_dict=urm, URM_mappers_dict=urm_mappers,
                            ICM_dict=folds[i].get_ICM_dict(), ICM_mappers_dict=folds[i].get_ICM_mappers_dict(),
                            UCM_dict=folds[i].get_UCM_dict(), UCM_mappers_dict=folds[i].get_UCM_mappers_dict())

            urm = {}
            test_urm = folds[i].get_URM()
            test_urm.sort_indices()
            mask = test_urm.data <= self.test_rating_threshold
            for URM_name in folds[i].get_URM_names():
                urm[URM_name] = folds[i].get_URM(URM_name)
                urm[URM_name].sort_indices()
                urm[URM_name].data[mask] = 0.0
                urm[URM_name].eliminate_zeros()

            test = Dataset(folds[i].get_name(), base_folder=folds[i].get_base_folder(),
                           postprocessings=folds[i].get_postprocessings(),
                           URM_dict=urm, URM_mappers_dict=folds[i].get_URM_mappers_dict(),
                           ICM_dict=folds[i].get_ICM_dict(), ICM_mappers_dict=folds[i].get_ICM_mappers_dict(),
                           UCM_dict=folds[i].get_UCM_dict(), UCM_mappers_dict=folds[i].get_UCM_mappers_dict())

            if not self.allow_cold_users:
                users_to_remove = np.arange(train.n_users)[np.ediff1d(train.get_URM().tocsr().indptr) <= 0]
                train.remove_users(users_to_remove)
                test.remove_users(users_to_remove)

            r.append((train, test))
        return r


####################
# isnt it the same as warm??
# maybe delete it
####################
class ColdItemsKFold(KFold):
    """
    The splitter tries to load from the specific folder related to a dataset, a split in the format corresponding to
    the splitter class. Basically each split is in a different subfolder
    - The "original" subfolder contains the whole dataset, is composed by a single URM with all data and may contain
        ICMs as well, either one or many, depending on the dataset
    - The other subfolders "warm", "cold" ecc contains the splitted data.

    The dataReader class involvement is limited to the following cased:
    - At first the dataSplitter tries to load from the subfolder corresponding to that split. Say "warm"
    - If the dataReader is succesful in loading the files, then a split already exists and the loading is complete
    - If the dataReader raises a FileNotFoundException, then no split is available.
    - The dataSplitter then creates a new instance of dataReader using default parameters, so that the original data will be loaded
    - At this point the chosen dataSplitter takes the URM_all and selected ICM to perform the split
    - The dataSplitter saves the splitted data in the appropriate subfolder.
    - Finally, the dataReader is instantiated again with the correct parameters, to load the data just saved
    """

    def get_name(self):
        return "cold_items_{}_fold_testthreshold_{:.1f}{}".format(self.n_folds, self.test_rating_threshold,
                                                                  "" if self.allow_cold_users else "_no_cold_users")

    def split(self, dataset):

        super(ColdItemsKFold, self).split(dataset)

        folds = []
        split_belonging = np.random.choice(self.n_folds, dataset.n_items, replace=True)

        for i in range(self.n_folds):

            urm = {}
            urm_mappers = {}
            mask = split_belonging != i
            for URM_name in dataset.get_URM_names():
                URM = dataset.get_URM(URM_name).tocsc(copy=True)
                # Sort nnz values by row and column indices, in order to remain consistent in the splits of different URMs
                for j in np.arange(URM.shape[1])[mask].tolist():
                    URM.data[URM.indptr[j]:URM.indptr[j + 1]] = 0.0
                URM.eliminate_zeros()
                urm[URM_name] = URM.tocsr()
                urm_mappers[URM_name] = dataset.get_URM_mapper(URM_name)

            folds.append(
                Dataset(dataset.get_name(), base_folder=dataset.get_base_folder(),
                        postprocessings=dataset.get_postprocessings(),
                        URM_dict=urm, URM_mappers_dict=urm_mappers,
                        ICM_dict=dataset.get_ICM_dict(), ICM_mappers_dict=dataset.get_ICM_mappers_dict(),
                        UCM_dict=dataset.get_UCM_dict(), UCM_mappers_dict=dataset.get_UCM_mappers_dict()
                        )
            )

        r = []
        for i in range(self.n_folds):
            urm = {}
            urm_mappers = {}
            for URM_name in folds[i].get_URM_names():
                # Keep i-th fold as test and merge the others as train
                urm[URM_name] = folds[(i + 1) % self.n_folds].get_URM(URM_name)
                urm_mappers[URM_name] = folds[(i + 1) % self.n_folds].get_URM_mapper(URM_name)
                for j in range(2, self.n_folds):
                    urm[URM_name] += folds[(i + j) % self.n_folds].get_URM(URM_name)

            train = Dataset(folds[i].get_name(), base_folder=folds[i].get_base_folder(),
                            postprocessings=folds[i].get_postprocessings(),
                            URM_dict=urm, URM_mappers_dict=urm_mappers,
                            ICM_dict=folds[i].get_ICM_dict(), ICM_mappers_dict=folds[i].get_ICM_mappers_dict(),
                            UCM_dict=folds[i].get_UCM_dict(), UCM_mappers_dict=folds[i].get_UCM_mappers_dict())

            urm = {}
            test_urm = folds[i].get_URM()
            test_urm.sort_indices()
            mask = test_urm.data <= self.test_rating_threshold
            for URM_name in folds[i].get_URM_names():
                urm[URM_name] = folds[i].get_URM(URM_name)
                urm[URM_name].sort_indices()
                urm[URM_name].data[mask] = 0.0
                urm[URM_name].eliminate_zeros()

            test = Dataset(folds[i].get_name(), base_folder=folds[i].get_base_folder(),
                           postprocessings=folds[i].get_postprocessings(),
                           URM_dict=urm, URM_mappers_dict=folds[i].get_URM_mappers_dict(),
                           ICM_dict=folds[i].get_ICM_dict(), ICM_mappers_dict=folds[i].get_ICM_mappers_dict(),
                           UCM_dict=folds[i].get_UCM_dict(), UCM_mappers_dict=folds[i].get_UCM_mappers_dict())

            if not self.allow_cold_users:
                users_to_remove = np.arange(train.n_users)[np.ediff1d(train.get_URM().tocsr().indptr) <= 0]
                train.remove_users(users_to_remove)
                test.remove_users(users_to_remove)

            r.append((train, test))

        return r
