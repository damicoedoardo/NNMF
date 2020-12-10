import pickle
# from Data_manager.load_and_save_data import save_data_dict, load_data_dict
# from Data_manager.split_functions.split_train_validation import split_train_validation_percentage_random_holdout
# from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix
import numpy as np
import scipy.sparse as sps
import zipfile
import os

from RecSysFramework.DataManager.Reader import DataReader
from RecSysFramework.DataManager.Utils import downloadFromURL, load_CSV_into_SparseBuilder

from RecSysFramework.DataManager import Dataset

'''
Created on Fri Nov 08 2019

@author XXX, XXX
'''


class PinterestReader(DataReader):

    DATASET_URL = "https://www.dropbox.com/s/op7m0ykdvtu0aub/pinterest-20.zip?dl=0"
    DATASET_SUBFOLDER = "Pinterest/"

    def __init__(self, reload_from_original_data=False):
        super(PinterestReader, self).__init__(reload_from_original_data)

    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER

    def _load_from_original_file(self):

        print("PinterestReader: Loading original data")

        zipFile_path = self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:
            dataFile = zipfile.ZipFile(zipFile_path + "pinterest-20.zip")

        except (FileNotFoundError, zipfile.BadZipFile):
            # print("PinterestReader: Unable to find data zip file.")
            # print("PinterestReader: Automatic download not available, please ensure the compressed data file"
            #       " is in folder {}.".format(zipFile_path))
            #print("PinterestReader: Data can be downloaded here: {}".format(
                 #self.DATASET_URL))

            # changed to enable automatic download
            downloadFromURL('https://uc3ad24dab7f41f66c947b62b23e.dl.dropboxusercontent.com/cd/0/get/AzgcTReKm5-lFSFIEuvYlWvmX7W0t4SvQBUsZhom8c_wY5AI7hE_8aXBBx6us8RkVh3XwyIGxftPNM0liwDzwC08wkuKkFApOtMXEB7EdMkA_HIuXTS03b5ieKhkK9yD4eo/file?_download_id=08845508258130064772350622594583565569370462589242554439469381047&_notify_doma',
                            zipFile_path, "pinterest-20.zip")
            dataFile = zipfile.ZipFile(zipFile_path + "pinterest-20.zip")

        URM_train_path = dataFile.extract(
            "pinterest-20.train.rating.txt", path=zipFile_path + "decompressed/")
        URM_test_path = dataFile.extract(
            "pinterest-20.test.rating.txt", path=zipFile_path + "decompressed/")
        trainMatrix = self.load_rating_file_as_matrix(URM_train_path)
        testRatings = self.load_rating_file_as_matrix(URM_test_path)

        from RecSysFramework.Utils.Common import reshapeSparse

        URM_train = trainMatrix.tocsr()
        URM_test = testRatings.tocsr()

        shape = (max(URM_train.shape[0], URM_test.shape[0]),
                 max(URM_train.shape[1], URM_test.shape[1]))

        URM_train = reshapeSparse(URM_train, shape)
        URM_test = reshapeSparse(URM_test, shape)

        mapper_users = {str(i+1): i for i in range(URM_train.shape[0])}
        mapper_items = {str(i+1): i for i in range(URM_train.shape[1])}

        return Dataset('Pinterest', URM_dict={"URM_all": URM_train+URM_test}, URM_mappers_dict={"URM_all": (mapper_users.copy(), mapper_items.copy())})

    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1:]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sps.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    mat[user, item] = 1.0
                line = f.readline()
        return mat

if __name__ == '__main__':
    r = PinterestReader()
    r._load_from_original_file()
