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


def avm_scoring(estimator, df):
    'return error from using fitted estimator with test data in the dataframe'
    assert isinstance(estimator, AVM)
    X, y = estimator.extract_and_transform(df)
    assert len(y) > 0
    y_hat = estimator.predict(df)
    errors = y_hat - y
    median_abs_error = np.median(np.abs(errors))
    return -median_abs_error  # because GridSearchCV chooses the model with the score


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

    cv = [(make_training_indices(samples, time_periods, t),
           make_testing_indices(samples, time_periods, t),
           )
          for t in xrange(len(time_periods) - 1)
          ]

#     AM's version of logic just above
#     cv = [df[df.yyyymm.isin(time_periods[:t+1])].index,
#           df[df.yyyymm.isin(time_periods[t+1])].index)
#           for t in len(time_periods)]
    # AM: INSTEAD
    # call GridSearchCV, specify cv object as
    # ((train_indices, test_indices), ...) for each slice of the time period
    # call fit(df)
    # implement AMV.score(test_part_of_df): run fitted model (self) and determine error (e.g. - L2)
    pdb.set_trace()
    # TODO: Review params with AM
    gscv = sklearn.grid_search.GridSearchCV(
        estimator=AVM(),
        param_grid=param_grid,
        scoring=avm_scoring,
        n_jobs=1 if control.test else -1,
        cv=cv,
        verbose=2 if control.test else 0,
    )
    # TODO AM: Can we first just validate and then run cross validation on the N best hyperparameter
    # settings
    print 'gscv'
    pprint(gscv)

    gscv.fit(samples)

    with open(control.path_out, 'wb') as f:
        pickle.dump((gscv, control), f)

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
