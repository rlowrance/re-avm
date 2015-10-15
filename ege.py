'''program to estimate the generalization error from a variety of AVMs

INPUT FILE:
    WORKING/samples-train-validate.csv
OUTPUT FILE:
    WORKING/ege.pickle
'''

from __future__ import division

import cPickle as pickle
import math
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
from Features import Features
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
    print 'usage  : python ege.py [--test]'
    print ' --test: run in test mode'
    sys.exit(1)


def make_control(argv):
    # return a Bunch

    print argv
    if len(argv) not in (1, 2):
        usage('invalid number of arguments')

    pcl = ParseCommandLine(argv)
    arg = Bunch(
        base_name=argv[0].split('.')[0],
        test=pcl.has_arg('--test'),
    )

    random_seed = 123
    random.seed(random_seed)

    dir_working = Path().dir_working()

    debug = False

    out_file_name_base = ('testing-' if arg.test else '') + arg.base_name

    return Bunch(
        arg=arg,
        debug=debug,
        path_in=dir_working + 'samples-train-validate.csv',
        path_out=dir_working + out_file_name_base + '.pickle',
        random_seed=random_seed,
        test=arg.test,
    )


def in_time_periods_function(df, time_periods):
    'return samples in any of the time periods'
    # NOTE: works only for monthly time periods coded yyyymm
    verbose = True
    sale_yyyymm = df[transactions.yyyymm]
    m = []
    for time_period in time_periods:
        m.append(sale_yyyymm == time_period)
    mask = reduce(lambda a, b: a | b, m) if len(m) > 1 else m[0]
    if verbose:
        print 'in_time_periods_function:'
        print ' time_periods', time_periods
        print ' n selected', sum(mask)
    return df.loc[mask]


def transformer_function(df, units):
    'return transformed X and y'
    pdb.set_trace()
    assert units in ('natural', 'log')
    pass


def make_X_y_function(df, time_period, X_units='log', y_units='natural'):
    'extract features X and target y from data frame'
    # TODO: build the age-related features
    # TODO: pass actual units in, instead of having caller take the default values
    print df.shape, time_period, X_units, y_units
    pdb.set_trace()
    features = Features().ege()
    X_transposed = np.array(len(features), len(df))
    for i, feature_transform in enumerate(features):
        feature, transform = feature_transform
        X_transposed[i] = df[feature] if X_units == 'natural' else df[feature].log()
    X = X_transposed.T
    y = np.array(len(df))
    for j in xrange(len(df)):
        y[j] = df[transactions.price][i] if y_units == 'natural' else math.log(df[transactions.price[i]])
    pdb.set_trace()
    return X, y


def using_time_series_cv(samples, control):
    # this function hold dead code
    def scoring_function():
        pass

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
        seq_units_X = ('log',)
        seq_units_y = ('natural',)

    common_params = {
        'n_months_back': seq_n_months_back,
    }
    param_grid = (
        common_params.copy.update(
            {'model_family': ['elastic_net'],
             'units_x': seq_units_X,
             'units_y': seq_units_y,
             'alpha': seq_alpha,
             'l1_ratio': seq_l1_ratio,
             'in_period': [in_time_periods_function],
             'scoring': [scoring_function],
             'n_months_back': seq_n_months_back,
             'transformer': [transformer_function],
             }
        ),
        common_params.copy.update(
            {'model_family': ['random_forest'],
             'units_x': ['natural'],
             'unitx_y': ['natural'],
             'n_estimators': seq_n_estimators,
             'max_depth': seq_max_depth,
             'in_period': [in_time_periods_function],
             'scoring': [scoring_function],
             'n_months_back': seq_n_months_back,
             'transformer': [transformer_function],
             }
        ),
    )

    pg = sklearn.grid_search.ParameterGrid(param_grid)
    print 'len(pg)', len(pg)

    time_periods = (
        200401, 200402, 200403, 200404, 200405, 200406, 200407, 200408, 200409, 200410, 200411, 200412,
        200501, 200502, 200503, 200504, 200505, 200506, 200507, 200508, 200509, 200510, 200511, 200512,
        200601, 200602, 200603, 200604, 200605, 200606, 200607, 200608, 200609, 200610, 200611, 200612,
        200701, 200702, 200703, 200704, 200705, 200706, 200707, 200708, 200709, 200710, 200711, 200712,
        200801, 200802, 200803, 200804, 200805, 200806, 200807, 200808, 200809, 200810, 200811, 200812,
        200901, 200902, 200903,
    )

    pdb.set_trace()

    def make_training_indices(df, t):
        'return 1D nparray of training indices in df'
        pdb.set_trace()
        reduced_df = df[t.yyyymm == time_periods[:(t + 1)]]
        result = reduced_df.index
        return result

    def make_testing_indices(df, t):
        'return 1D nparray of testing indices in df'
        pdb.set_trace()
        pass

    cv = [(make_training_indices(samples, t), make_testing_indices(samples, t))
          for t in len(time_periods)
          ]
#     cv = [df[df.yyyymm.isin(time_periods[:t+1])].index,
#           df[df.yyyymm.isin(time_periods[t+1])].index)
#           for t in len(time_periods)]
    # AM: INSTEAD
    # call GridSearchCV, specify cv object as
    # ((train_indices, test_indices), ...) for each slice of the time period
    # call fit(df)
    # implement AMV.score(test_part_of_df): run fitted model (self) and determine error (e.g. - L2)
    gscb = sklearn.grid_search.GridSearchCV(
        estimator=AVM(),
        param_grid=param_grid,
        scoring=AVM.score,
        n_jobs=1 if control.test else -1,
        cv=cv,
        verbose=2 if control.test else 0,
    )
    print 'gscb'
    pprint(gscb)

    gscb.fit(samples)

#     tscv = TimeSeriesCV(
#         estimator=AVM(),
#         param_grid_model_search=param_grid_model_search,
#         scoring=scoring_function,
#         time_periods=time_periods,
#         in_time_periods=in_time_periods_function,
#         make_X_y=make_X_y_function,
#         test=control.test,
#         verbose=1,
#     )
#
#     tscv.fit(samples)
#     pprint(tscv)


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

    scoring_function = sklearn.metrics.make_scorer(
        sklearn.metrics.mean_squared_error,
        greater_is_better=False,
    )

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
    }
    elastic_net_params = common_params.copy()
    elastic_net_params.update(
        {'model_family': ['elastic_net'],
         'units_x': seq_units_X,
         'units_y': seq_units_y,
         'alpha': seq_alpha,
         'l1_ratio': seq_l1_ratio,
         'in_period': [in_time_periods_function],
         'scoring': [scoring_function],
         'n_months_back': seq_n_months_back,
         'transformer': [transformer_function],
         }
    )
    random_forest_params = common_params.copy()
    random_forest_params.update(
        {'model_family': ['random_forest'],
         'units_x': ['natural'],
         'unitx_y': ['natural'],
         'n_estimators': seq_n_estimators,
         'max_depth': seq_max_depth,
         'in_period': [in_time_periods_function],
         'scoring': [scoring_function],
         'n_months_back': seq_n_months_back,
         'transformer': [transformer_function],
         }
    )
    param_grid = (
        elastic_net_params,
        random_forest_params,
    )

    pg = sklearn.grid_search.ParameterGrid(param_grid)
    print 'len(pg)', len(pg)

    time_periods = (
        200401, 200402, 200403, 200404, 200405, 200406, 200407, 200408, 200409, 200410, 200411, 200412,
        200501, 200502, 200503, 200504, 200505, 200506, 200507, 200508, 200509, 200510, 200511, 200512,
        200601, 200602, 200603, 200604, 200605, 200606, 200607, 200608, 200609, 200610, 200611, 200612,
        200701, 200702, 200703, 200704, 200705, 200706, 200707, 200708, 200709, 200710, 200711, 200712,
        200801, 200802, 200803, 200804, 200805, 200806, 200807, 200808, 200809, 200810, 200811, 200812,
        200901, 200902, 200903,
    )

    def selected_indices(values, df):
        mask = df[transactions.yyyymm].isin(values)
        result = mask[mask].index
        return result

    def make_training_indices(df, t):
        'return 1D nparray of training indices in df'
        values = time_periods[:(t + 1)]
        return selected_indices(values, df)

    def make_testing_indices(df, t):
        'return 1D nparray of testing indices in df'
        values = (time_periods[t + 1],)  # must pass an iterable to selected_indices
        return selected_indices(values, df)

    pdb.set_trace()
    cv = [(make_training_indices(samples, t), make_testing_indices(samples, t))
          for t in xrange(len(time_periods))
          ]

#     cv = [df[df.yyyymm.isin(time_periods[:t+1])].index,
#           df[df.yyyymm.isin(time_periods[t+1])].index)
#           for t in len(time_periods)]
    # AM: INSTEAD
    # call GridSearchCV, specify cv object as
    # ((train_indices, test_indices), ...) for each slice of the time period
    # call fit(df)
    # implement AMV.score(test_part_of_df): run fitted model (self) and determine error (e.g. - L2)
    gscb = sklearn.grid_search.GridSearchCV(
        estimator=AVM(),
        param_grid=param_grid,
        scoring=AVM.score,
        n_jobs=1 if control.test else -1,
        cv=cv,
        verbose=2 if control.test else 0,
    )
    print 'gscb'
    pprint(gscb)

    gscb.fit(samples)

    with open(control.path_out, 'wb') as f:
        pickle.dump((gscb, control), f)

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
