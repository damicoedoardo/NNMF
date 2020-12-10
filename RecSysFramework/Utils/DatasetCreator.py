from RecSysFramework.DataManager.Reader.Movielens1MReader import Movielens1MReader
from RecSysFramework.DataManager.Reader.LastFMHetrec2011Reader import LastFMHetrec2011Reader
from RecSysFramework.DataManager.Reader.BookCrossingReader import BookCrossingReader
from RecSysFramework.DataManager.Reader.CiteULikeReader import CiteULike_aReader
from RecSysFramework.DataManager.Splitter import Holdout
from RecSysFramework.DataManager.Splitter import KFold
from RecSysFramework.DataManager.DatasetPostprocessing.ImplicitURM import ImplicitURM
# from RecSysFramework.DataManager.DatasetPostprocessing.LongQueueAnalysis import LongQueueAnalysis
from RecSysFramework.DataManager.DatasetPostprocessing.KCore import KCore
from RecSysFramework.DataManager.Splitter.KFold import WarmItemsKFold
import RecSysFramework.Utils.menu as menu
from RecSysFramework.DataManager.Reader.PinterestReader import PinterestReader
import random

def create_dataset(save_folder_path=None, random_seed=True):
    """
    Create, split and save datasets.
    """
    if random_seed == True:
        random_seed = random.randint(1,500)
    else:
        random_seed = 42
    print(random_seed)
    dataset_name = menu.single_choice('Select the dataset you want to create',
                       ['Movielens1MReader', 'LastFMHetrec2011Reader', 'BookCrossingReader', 'CiteULike_aReader', 'PinterestReader'])

    dataset_reader = eval(dataset_name)()
    dataset = dataset_reader.load_data()

    implicitization = menu.yesno_choice('Perform Implicitization?')

    if implicitization == 'y':
        threshold = float(input('insert the threshold for Implicitization\n'))
        assert threshold <= dataset.get_URM().data.max(), 'selected threshold is too high!'
        implicitizer = ImplicitURM(int(threshold))
        dataset = implicitizer.apply(dataset)
        print('implicitization complete!')

    lq_analysis = menu.yesno_choice('Perform Long Queue Analisys?')
    if lq_analysis == 'y':
        lqa = LongQueueAnalysis()
        dataset = lqa.apply(dataset)

    kcore = menu.yesno_choice('Perform KCore postprocessing ?')

    if kcore == 'y':
        user_K = int(input('insert USER K\n'))
        item_K = int(input('insert ITEM K\n'))
        kcore = KCore(user_k_core=user_K, item_k_core=item_K)
        dataset = kcore.apply(dataset)

    dataset_name = menu.single_choice('which kind of splitting?',
                       ['Holdout', 'KFold'])

    if dataset_name == 'Holdout':
        print('NOTE the percentage have to sum to 1\n')
        train_perc = float(input('insert train percentage\n'))
        val_perc = float(input('insert validation percentage\n'))
        test_perc = float(input('insert test percentage\n'))
        splitter = Holdout(train_perc=train_perc, validation_perc=val_perc, test_perc=test_perc, random_seed=random_seed)
    elif dataset_name == 'KFold':
        n_folds = int(input('how many folds?\n'))
        train_perc = float(input('which percentage of the dataset you want to divide in {} folds?\n'.format(n_folds)))
        splitter = WarmItemsKFold(n_folds=n_folds, percentage_initial_data_to_split=train_perc, random_seed=random_seed)

    save_dataset = menu.yesno_choice('Save the Dataset?\n')
    if save_dataset == 'y':
        splitter.save_split(splitter.split(dataset), filename_suffix='', save_folder_path=save_folder_path)

if __name__ == '__main__':
    create_dataset(random_seed=False)
