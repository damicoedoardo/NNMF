
from RecSysFramework.Utils import menu
from RecSysFramework.DataManager.Reader.Movielens1MReader import Movielens1MReader
from RecSysFramework.DataManager.Reader.LastFMHetrec2011Reader import LastFMHetrec2011Reader
from RecSysFramework.DataManager.Reader.BookCrossingReader import BookCrossingReader
from RecSysFramework.DataManager.DatasetPostprocessing.ImplicitURM import ImplicitURM
from RecSysFramework.DataManager.DatasetPostprocessing.KCore import KCore
from RecSysFramework.DataManager.DatasetPostprocessing.LongQueueAnalysis import LongQueueAnalysis
from RecSysFramework.DataManager.Splitter import Holdout
from RecSysFramework.DataManager.Reader.CiteULikeReader import CiteULike_aReader
from RecSysFramework.DataManager.Reader.PinterestReader import PinterestReader


def retrieve_train_validation_test_holdhout_dataset(dataset_name=None):
    if dataset_name == None:
        dataset_name = menu.single_choice('Select the dataset you want to create',
                                          ['Movielens1MReader', 'LastFMHetrec2011Reader', 'BookCrossingReader',
                                           'CiteULike_aReader', 'PinterestReader'])

    if dataset_name == 'Movielens1MReader':
        reader = Movielens1MReader()
        h = Holdout(train_perc=0.6, validation_perc=0.2, test_perc=0.2)
        train, test, validation = h.load_split(
            reader, postprocessings=[ImplicitURM(3), KCore(5, 5)])
    if dataset_name == 'BookCrossingReader':
        reader = BookCrossingReader()
        h = Holdout(train_perc=0.6, validation_perc=0.2, test_perc=0.2)
        train, test, validation = h.load_split(
            reader, postprocessings=[ImplicitURM(6), KCore(5, 5)])
    if dataset_name == 'CiteULike_aReader':
        reader = CiteULike_aReader()
        h = Holdout(train_perc=0.6, validation_perc=0.2, test_perc=0.2)
        train, test, validation = h.load_split(
            reader, postprocessings=[KCore(5, 5)])
    if dataset_name == 'LastFMHetrec2011Reader':
        reader = LastFMHetrec2011Reader()
        h = Holdout(train_perc=0.6, validation_perc=0.2, test_perc=0.2)
        train, test, validation = h.load_split(
            reader, postprocessings=[ImplicitURM(1), KCore(5, 5)])
    if dataset_name == 'PinterestReader':
        reader = PinterestReader()
        h = Holdout(train_perc=0.6, validation_perc=0.2, test_perc=0.2)
        train, test, validation = h.load_split(
            reader, postprocessings=[ImplicitURM(1), KCore(5, 5)])

    return train, test, validation, dataset_name.replace('Reader', '').replace('CiteULike_a', 'CiteULike-a')
