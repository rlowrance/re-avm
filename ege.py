'''program to estimate the generalization error from a variety of AVMs

INPUT FILE:
    WORKING/samples-train-validate.csv
OUTPUT FILE:
    WORKING/ege.pickle
'''

from __future__ import division

import cPickle as pickle
import numpy as np
import pandas as pd
import pdb
from pprint import pprint
import random
import sklearn
import sklearn.grid_search
import sklearn.metrics
import sys

from AVM import AVM
from Bunch import Bunch
from columns_contain import columns_contain
import layout_transactions as transactions
from Logger import Logger
from MonthSelector import MonthSelector
from ParseCommandLine import ParseCommandLine
from Path import Path
# from TimeSeriesCV import TimeSeriesCV
cc = columns_contain


def usage(msg=None):
    if msg is not None:
        print msg
    print 'usage  : python ege.py [--rfbound] [--test]'
    print ' --rfbound: only determine bounds on RF hyperparameters'
    print ' --test: run in test mode'
    sys.exit(1)


def make_control(argv):
    # return a Bunch

    print argv
    if len(argv) not in (1, 2, 3):
        usage('invalid number of arguments')

    pcl = ParseCommandLine(argv)
    arg = Bunch(
        base_name=argv[0].split('.')[0],
        test=pcl.has_arg('--test'),
        rfbound=pcl.has_arg('--rfbound'),
    )

    random_seed = 123
    random.seed(random_seed)

    dir_working = Path().dir_working()

    debug = False

    out_file_name_base = ('testing-' if arg.test else '') + arg.base_name

    return Bunch(
        arg=arg,
        debug=debug,
        n_cv_folds=10,
        path_in=dir_working + 'samples-train-validate.csv',
        path_out=dir_working + out_file_name_base + '.pickle',
        random_seed=random_seed,
        test=arg.test,
    )


def make_time_series_cv_folds(samples, time_periods):
    'return folds needed if we are doing time-series cross validation'

    def make_training_indices(df, time_periods, t):
        'return vector of training indices in df for time_periods[0 .. t + 1]'
        mask = df[transactions.yyyymm].isin(time_periods[:(t + 1)])
        result = df.index[mask]
        return result

    def make_testing_indices(df, time_periods, t):
        'return vector of testing indices in df for time_period[t + 1]'
        mask = df[transactions.yyyymm].isin((time_periods[t + 1],))
        result = df.index[mask]
        return result
#     AM's version of logic below
#     cv = [df[df.yyyymm.isin(time_periods[:t+1])].index,
#           df[df.yyyymm.isin(time_periods[t+1])].index)
#           for t in len(time_periods)]
    folds = [(make_training_indices(samples, time_periods, t),
              make_testing_indices(samples, time_periods, t),
              )
             for t in xrange(len(time_periods) - 1)
             ]
    return folds


def avm_scoring(estimator, df):
    'return error from using fitted estimator with test data in the dataframe'
    # TODO: make a static method of class AVM
    assert isinstance(estimator, AVM)
    X, y = estimator.extract_and_transform(df)
    assert len(y) > 0
    y_hat = estimator.predict(df)
    errors = y_hat - y
    median_abs_error = np.median(np.abs(errors))
    return -median_abs_error  # because GridSearchCV chooses the model with the score


def print_gscv(gscv, tag=None, only_best=False):
    print 'result from GridSearchCV'
    if tag is not None:
        print 'for', str(tag)

    def print_params(params):
        for k, v in params.iteritems():
            print ' parameter %15s: %s' % (k, v)

    def print_grid_score(gs):
        print ' mean: %.0f std: %0.f' % (gs.mean_validation_score, np.std(gs.cv_validation_scores))
        for cv_vs in gs.cv_validation_scores:
            print ' validation score: %0.6f' % cv_vs
        print_params(gs.parameters)

    if not only_best:
        for i, grid_score in enumerate(gscv.grid_scores_):
            print 'grid index', i
            print_grid_score(grid_score)
    print 'best score', gscv.best_score_
    print 'best estimator', gscv.best_estimator_
    print 'best params'
    print_params(gscv.best_params_)
    print 'scorer', gscv.scorer_


def do_normal_run(control, samples):
    'cross validate to find the best model'
    # hyperparameter search grid (across both linear and tree model forms)
    seq_alpha = (0.1, 0.3, 1.0, 3.0, 10.0)
    seq_l1_ratio = (0.0, 0.2, 0.4, 0.8, 1.0)
    seq_max_depth = (1, 3, 10, 30, 100),
    seq_n_estimators = (100, 300, 1000)
    seq_n_months_back = (1, 2, 3, 4, 5, 6)
    seq_units_X = ('natural', 'log')
    seq_units_y = ('natural', 'log')

    if control.test:
        seq_alpha = (3.0,)
        seq_l1_ratio = (0.4,)
        seq_max_depth = (10,)
        seq_n_estimators = (100,)
        seq_n_months_back = (2,)
        seq_units_X = ('log',)
        seq_units_y = ('natural',)

    common_params = {
        'n_months_back': seq_n_months_back,
        'random_state': (control.random_seed,),
    }
    elastic_net_params = common_params.copy()
    elastic_net_params.update(
        {'model_name': ['ElasticNet'],
         'units_X': seq_units_X,
         'units_y': seq_units_y,
         'alpha': seq_alpha,
         'l1_ratio': seq_l1_ratio,
         }
    )
    random_forest_params = common_params.copy()
    random_forest_params.update(
        {'model_name': ['RandomForestRegressor'],
         'n_estimators': seq_n_estimators,
         'max_depth': seq_max_depth,
         }
    )
    param_grid = (
        elastic_net_params,
        random_forest_params,
    )

    pg = sklearn.grid_search.ParameterGrid(param_grid)
    print 'param_grid'
    pprint(param_grid)
    print 'len(pg)', len(pg)

    time_periods = (
        200401, 200402, 200403, 200404, 200405, 200406, 200407, 200408, 200409, 200410, 200411, 200412,
        200501, 200502, 200503, 200504, 200505, 200506, 200507, 200508, 200509, 200510, 200511, 200512,
        200601, 200602, 200603, 200604, 200605, 200606, 200607, 200608, 200609, 200610, 200611, 200612,
        200701, 200702, 200703, 200704, 200705, 200706, 200707, 200708, 200709, 200710, 200711, 200712,
        200801, 200802, 200803, 200804, 200805, 200806, 200807, 200808, 200809, 200810, 200811, 200812,
        200901, 200902, 200903,
    )

    if False:
        time_series_cv_folds = make_time_series_cv_folds(samples, time_periods)
        pprint(time_series_cv_folds)

    pdb.set_trace()
    # TODO: Review params with AM
    gscv = sklearn.grid_search.GridSearchCV(
        estimator=AVM(),
        param_grid=param_grid,
        scoring=avm_scoring,
        n_jobs=1 if control.test else -1,
        cv=control.n_cv_folds,
        verbose=2 if control.test else 0,
    )
    # TODO AM: Can we first just validate and then run cross validation on the N best hyperparameter
    # settings
    # OR run with cv=2, then run with cv=10 on best ~10 models
    print 'gscv before fitting'
    pprint(gscv)

    gscv.fit(samples)
    print
    print_gscv(gscv)

    return gscv


def do_rfbound(control, samples):
    'determine reasonable bounds on Random Forests HPs n_estimators, n_trees'

    # HP settings to test
    model_name_seq = ('RandomForestRegressor',)
    n_months_back_seq = (1, 2, 3, 4, 5, 6)
    forecast_time_period_seq = (200901, 200902, 200903)
    n_estimators_seq = (10, 30, 100, 300, 1000)
    max_depth_seq = (1, 3, 10, 30, 100, 300)

    results = {}
    for forecast_time_period in forecast_time_period_seq:
        gscv = sklearn.grid_search.GridSearchCV(
            estimator=AVM(),
            param_grid=dict(
                model_name=model_name_seq,
                n_months_back=n_months_back_seq,
                forecast_time_period=[forecast_time_period],
                n_estimators=n_estimators_seq,
                max_depth=max_depth_seq,
                random_state=[control.random_seed],
            ),
            scoring=avm_scoring,
            n_jobs=1 if control.test else -1,
            cv=2 if control.test else control.n_cv_folds,
            verbose=0 if control.test else 0,
        )
        gscv.fit(samples)
        print
        print_gscv(gscv, tag=str(forecast_time_period), only_best=True)
        results[forecast_time_period] = gscv
    return results


def main(argv):
    control = make_control(argv)
    if False:
        # avoid error in sklearn that requires flush to have no arguments
        sys.stdout = Logger(base_name=control.arg.base_name)
    print control

    samples = pd.read_csv(
        control.path_in,
        nrows=1000 if control.test else None,
    )
    print 'samples.shape', samples.shape

    if control.arg.rfbound:
        result = do_rfbound(control, samples)
    else:
        result = do_normal_run(control, samples)

    with open(control.path_out, 'wb') as f:
        pickle.dump((result, control), f)

    print control
    if control.test:
        print 'DISCARD OUTPUT: test'
    print 'done'

    return


if __name__ == '__main__':
    if False:
        # avoid pyflakes warnings
        AVM()
        MonthSelector()
        pdb.set_trace()
        pprint()
        pd.DataFrame()
        np.array()
        print transactions

    main(sys.argv)
