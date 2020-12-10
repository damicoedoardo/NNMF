#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 16/09/19

@author: XXX
"""

from RecSysFramework.Recommender.DataIO import DataIO

import time
import os
import traceback
import numpy as np

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical

from RecSysFramework.Utils.EarlyStopping import EarlyStoppingModel

from .SearchAbstractClass import SearchAbstractClass, writeLog, get_result_string_evaluate_on_validation


class AbstractSearchIterationStrategy:

    def search_iteration(self, obj):
        pass


class SearchIterationStrategyRandom(AbstractSearchIterationStrategy):

    ITERATION_STRATEGY='random'

    def search_iteration(self, obj):
        for _ in range(obj.n_calls):
            # sample from params
            samples = []
            for v in obj.hyperparams_values:
                samples.append(v.rvs(n_samples=1)[0])

            # eval
            obj.result = obj._objective_function_list_input(samples)


class SearchIterationStrategyBayesianSkopt(AbstractSearchIterationStrategy):

    ITERATION_STRATEGY='bayesian_skopt'

    def __init__(self,
                 n_random_starts=20,
                 n_points=10000,
                 n_jobs=1,
                 # noise='gaussian',
                 noise=1e-5,
                 acq_func='gp_hedge',
                 acq_optimizer='auto',
                 random_state=None,
                 verbose=True,
                 n_restarts_optimizer=10,
                 xi=0.01,
                 kappa=1.96,
                 x0=None,
                 y0=None):
        """
        wrapper to change the params of the bayesian optimizator.
        for further details:
        https://scikit-optimize.github.io/#skopt.gp_minimize

        """
        self.n_point = n_points
        self.n_random_starts = n_random_starts
        self.n_jobs = n_jobs
        self.acq_func = acq_func
        self.acq_optimizer = acq_optimizer
        self.random_state = random_state
        self.n_restarts_optimizer = n_restarts_optimizer
        self.verbose = verbose
        self.xi = xi
        self.kappa = kappa
        self.noise = noise
        self.x0 = x0
        self.y0 = y0

    def search_iteration(self, obj):
        obj.result = gp_minimize(obj._objective_function_list_input,
                                 obj.hyperparams_values,
                                 base_estimator=None,
                                 n_calls=obj.n_calls,
                                 n_random_starts=self.n_random_starts,
                                 acq_func=self.acq_func,
                                 acq_optimizer=self.acq_optimizer,
                                 x0=self.x0,
                                 y0=self.y0,
                                 random_state=self.random_state,
                                 verbose=self.verbose,
                                 n_points=self.n_point,
                                 n_restarts_optimizer=self.n_restarts_optimizer,
                                 xi=self.xi,
                                 kappa=self.kappa,
                                 noise=self.noise,
                                 n_jobs=self.n_jobs)
