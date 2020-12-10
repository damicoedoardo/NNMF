from RecSysFramework.DataManager.DatasetPostprocessing.ImplicitURM import ImplicitURM
from RecSysFramework.DataManager.DatasetPostprocessing.LongQueueAnalysis import LongQueueAnalysis
from RecSysFramework.DataManager.DatasetPostprocessing.KCore import KCore
from RecSysFramework.DataManager.Splitter import Holdout
from RecSysFramework.Evaluation.Evaluator import EvaluatorHoldout
from RecSysFramework.ParameterTuning.SearchOneFold import SearchOneFold
from RecSysFramework.Recommender.SLIM.BPR.SLIM_BPR_Cython import SLIM_BPR_Cython
from RecSysFramework.ParameterTuning.SearchIterationStrategy import SearchIterationStrategyBayesianSkopt
from RecSysFramework.ParameterTuning.SearchIterationStrategy import SearchIterationStrategyRandom
from RecSysFramework.Recommender.MatrixFactorization.MatrixFactorization_Cython import BPR_NNMF
from RecSysFramework.Recommender.MatrixFactorization.MatrixFactorization_Cython import PROB_NNMF
from RecSysFramework.Recommender.MatrixFactorization.MatrixFactorization_Cython import FUNK_NNMF
from RecSysFramework.ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from skopt.space import Real, Integer, Categorical
import RecSysFramework.Utils.menu as menu
import RecSysFramework.Utils.get_holdout as datasetreader

train, test, validation, dataset_name = datasetreader.retrieve_train_validation_test_holdhout_dataset()

# setting up evaluator
evalh = EvaluatorHoldout([5])
evalh.global_setup(URM_test=validation.get_URM())

recommender_input_args = SearchInputRecommenderArgs(CONSTRUCTOR_KEYWORD_ARGS={'URM_train': train.get_URM()},
                                                        FIT_KEYWORD_ARGS={
                                                            'validation_every_n': 15,
                                                            'sgd_mode': 'adam',
                                                            'lower_validations_allowed': 10,
                                                            'validation_metric': 'MAP',
                                                            'evaluator_object': evalh,
                                                            'stop_on_validation': True,
                                                            'verbose': False,
                                                            'epochs': 10000,
                                                            'symmetric': True,
                                                            'random_seed':None,
                                                            'positive_threshold_BPR':None,
                                                            'train_with_sparse_weights':None
                                                        })

space = {
        'batch_size': Categorical([1, 10, 100, 200, 350, 600, 1000, 1500, 2000, 3500, 5000, 7500, 10000]),
        'lambda_i':Real(0, 1),
        'lambda_j': Real(0, 1),
        'learning_rate': Real(1e-5, 1e-1),
        'topK': Integer(10,500),
        'gamma':Real(0,1),
        'beta_1':Real(0,1),
        'beta_2':Real(0,1),
         }

search_mode = menu.single_choice('Select optimizer', ['SearchIterationStrategyRandom', 'SearchIterationStrategyBayesianSkopt'])

optimizer = SearchOneFold(SLIM_BPR_Cython, eval(search_mode)(n_random_starts=30), evalh)

optimizer.search(recommender_input_args, space, save_metadata=False, save_model='best', n_cases=1000,
                 metric_to_optimize='MAP', telegram_bot=True, dataset_name=dataset_name)
