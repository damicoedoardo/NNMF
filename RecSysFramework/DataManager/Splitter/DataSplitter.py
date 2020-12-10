#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 30/11/18

@author: XXX, XXX
"""

import numpy as np
import os

from RecSysFramework.DataManager import Dataset
from RecSysFramework.DataManager.Reader import DataReader

class DataSplitter(object):


    def __init__(self, forbid_new_split=False, force_new_split=False, with_validation=True, allow_cold_users=False, random_seed=42):
        super(DataSplitter, self).__init__()
        self.forbid_new_split = forbid_new_split
        self.force_new_split = force_new_split
        self.with_validation = with_validation
        self.random_seed = random_seed
        self.allow_cold_users = allow_cold_users


    def _get_dataset_names_in_split(self):
        datalist = ["train", "test"]
        if self.with_validation:
            datalist.append("validation")
        return datalist


    def get_name(self):
        pass


    def split(self, dataset):
        np.random.seed(self.random_seed)


    def load_split(self, datareader, save_folder_path=None, postprocessings=None):

        if save_folder_path is None:
            tmp_save_folder_path = datareader.get_complete_default_save_path(postprocessings)
        else:
            tmp_save_folder_path = save_folder_path+os.sep

        try:

            datalist = self._get_dataset_names_in_split()
            for i in datalist:
                if not datareader.all_files_available(tmp_save_folder_path + self.get_name() + os.sep, filename_suffix="_{}".format(i)):
                    raise Exception

            datasets = []
            for i in datalist:
                urm, urm_mappers, icm, icm_mappers, ucm, ucm_mappers = datareader.load_from_saved_sparse_matrix(
                    tmp_save_folder_path + self.get_name() + os.sep, filename_suffix="_{}".format(i))
                datasets.append(Dataset(datareader.get_dataset_name(),
                                        base_folder=datareader.get_default_save_path(),
                                        postprocessings=postprocessings,
                                        URM_dict=urm, URM_mappers_dict=urm_mappers,
                                        ICM_dict=icm, ICM_mappers_dict=icm_mappers,
                                        UCM_dict=ucm, UCM_mappers_dict=ucm_mappers))

            return datasets

        except:

            print("DataSplitter: Preloaded data not found or corrupted, reading from original files...")
            dataset = datareader.load_data(save_folder_path=save_folder_path, postprocessings=postprocessings)
            return self.split(dataset)


    def save_split(self, split, save_folder_path=None, filename_suffix=""):
        dr = DataReader()

        datalist = self._get_dataset_names_in_split()
        assert len(split) == len(datalist), "DataSplitter: expected {} datasets in split, given {}".format(len(datalist), len(split))

        if save_folder_path is None:
            save_folder_path = split[0].get_complete_folder() + self.get_name() + os.sep
        else:
            save_folder_path = save_folder_path+os.sep+self.get_name()+os.sep
        for i, dataset in enumerate(split):
            dataset.save_data(save_folder_path=save_folder_path,
                              filename_suffix="{}_{}".format(filename_suffix, datalist[i]))
