'''Determine accuracy on validation set YYYYMM of various hyperparameter setting
for AVMs based on 3 models (linear, random forests, gradient boosting regression

INVOCATION
  python valavm.py YYYYMM [-test]

INPUT FILE:
  WORKING/samples-train-validate.csv

OUTPUT FILE:
  WORKING/valgrb/YYYYMM.pickle
'''

from __future__ import division

import collections
import cPickle as pickle
import numpy as np
import os
import pandas as pd
import pdb
from pprint import pprint
import random
import sys

import AVM
from Bunch import Bunch
from columns_contain import columns_contain
import layout_transactions
from Logger import Logger
from ParseCommandLine import ParseCommandLine
from Path import Path
# from TimeSeriesCV import TimeSeriesCV
cc = columns_contain


def usage(msg=None):
    print __doc__
    if msg is not None:
        print msg
    sys.exit(1)


def make_control(argv):
    # return a Bunch

    print argv
    if not(2 <= len(argv) <= 3):
        usage('invalid number of arguments')

    pcl = ParseCommandLine(argv)
    arg = Bunch(
        base_name='valavm',
        yyyymm=argv[1],
        test=pcl.has_arg('--test'),
    )

    try:
        arg.yyyymm = int(arg.yyyymm)
    except:
        usage('YYYYMM not an integer')

    random_seed = 123
    random.seed(random_seed)

    dir_working = Path().dir_working()

    debug = False

    out_file_name = (
        ('test-' if arg.test else '') +
        '%s.pickle' % arg.yyyymm
    )

    # assure output directory exists
    dir_path = dir_working + arg.base_name + '/'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    fixed_hps = Bunch(
        loss='quantile',
        alpha=0.5,
        n_estimators=1000,
        max_depth=3,
        max_features=None)

    return Bunch(
        arg=arg,
        debug=debug,
        fixed_hps=fixed_hps,
        path_in=dir_working + 'samples-train-validate.csv',
        path_out=dir_path + out_file_name,
        random_seed=random_seed,
        test=arg.test,
    )

ResultKeyEn = collections.namedtuple(
    'ResultKeyEn',
    'n_months_back units_X units_y alpha l1_ratio',
)
ResultKeyGbr = collections.namedtuple(
    'ResultKeyGbr',
    'n_months_back n_estimators max_features max_depth loss learning_rate',
)
ResultKeyRfr = collections.namedtuple(
    'ResultKeyRfr',
    'n_months_back n_estimators max_features max_depth',
)
ResultValue = collections.namedtuple(
    'ResultValue',
    'actuals predictions',
)


def do_val(control, samples):
    'run grid search on hyperparameters across the 3 model kinds'

    def check_for_missing_predictions(result):
        for k, v in result.iteritems():
            if v.predictions is None:
                print k
                print 'found missing predictions'
                pdb.set_trace()

    # HP settings to test across all models
    n_months_back_seq = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)

    # HP settings to test for ElasticNet models
    alpha_seq = (0.01, 0.03, 0.1, 0.3, 1.0)  # multiplies the penalty term
    l1_ratio_seq = (0.0, 0.25, 0.50, 0.75, 1.0)  # 0 ==> L2 penalty, 1 ==> L1 penalty
    units_X_seq = ('natural', 'log')
    units_y_seq = ('natural', 'log')

    # HP settings to test for tree-based models
    n_estimators_seq = (10, 30, 100, 300, 1000)
    max_features_seq = (1, 'log2', 'sqrt', .1, .3, 'auto')
    max_depth_seq = (1, 3, 10, 30, 100, 300)

    # reduce grid size to shorten computation time
    n_estimators_seq = (10, 30, 100, 300)
    max_features_seq = (1, 'log2', 'sqrt', 'auto')
    max_depth_seq = (1, 3, 10, 30, 100)

    # HP setting to test for GradientBoostingRegression models
    learning_rate_seq = (.10, .20, .30, .40, .50, .60, .70, .80, .90)
    loss_seq = ('ls', 'lad', 'quantile')

    # reduce grid size to shorten computation time
    learning_rate_seq = (.10, .25, .50, .75, .99)
    loss_seq = ('ls', 'quantile')

    def max_features_s(max_features):
        'convert to 4-character string (for printing)'
        return max_features[:4] if isinstance(max_features, str) else ('%4.1f' % max_features)

    result = {}

    def fit_and_run(avm):
        'return a ResultValue'
        avm.fit(samples)
        mask = samples[layout_transactions.yyyymm] == control.arg.yyyymm
        samples_yyyymm = samples[mask]
        predictions = avm.predict(samples_yyyymm)
        if predictions is None:
            print 'no predictions!'
            pdb.set_trace()
        actuals = samples_yyyymm[layout_transactions.price]
        return ResultValue(actuals, predictions)

    def search_en(n_months_back):
        'search over ElasticNet HPs, appending to result'
        for units_X in units_X_seq:
            for units_y in units_y_seq:
                for alpha in alpha_seq:
                    for l1_ratio in l1_ratio_seq:
                        print (
                            '%6d %3s %1d %3s %3s %4.2f %4.2f' %
                            (control.arg.yyyymm, 'en', n_months_back, units_X[:3], units_y[:3],
                             alpha, l1_ratio)
                        )
                        avm = AVM.AVM(
                            model_name='ElasticNet',
                            forecast_time_period=control.arg.yyyymm,
                            random_state=control.random_seed,
                            n_months_back=n_months_back,
                            units_X=units_X,
                            units_y=units_y,
                            alpha=alpha,
                            l1_ratio=l1_ratio,
                        )
                        result_key = ResultKeyEn(
                            n_months_back,
                            units_X,
                            units_y,
                            alpha,
                            l1_ratio,
                        )
                        result[result_key] = fit_and_run(avm)
                        if control.test:
                            return

    def search_gbr(n_months_back):
        'search over GradientBoostingRegressor HPs, appending to result'
        for n_estimators in n_estimators_seq:
            for max_features in max_features_seq:
                for max_depth in max_depth_seq:
                    for loss in loss_seq:
                        for learning_rate in learning_rate_seq:
                            print (
                                '%6d %3s %1d %4d %4s %3d %8s %4.2f' %
                                (control.arg.yyyymm, 'gbr', n_months_back,
                                 n_estimators, max_features_s(max_features), max_depth, loss, learning_rate)
                            )
                            avm = AVM.AVM(
                                model_name='GradientBoostingRegressor',
                                forecast_time_period=control.arg.yyyymm,
                                random_state=control.random_seed,
                                n_months_back=n_months_back,
                                learning_rate=learning_rate,
                                loss=loss,
                                alpha=.5 if loss == 'quantile' else None,
                                n_estimators=n_estimators,  # number of boosting stages
                                max_depth=max_depth,  # max depth of any tree
                                max_features=max_features,  # how many features to test when splitting
                            )
                            result_key = ResultKeyGbr(
                                n_months_back,
                                n_estimators,
                                max_features,
                                max_depth,
                                loss,
                                learning_rate,
                            )
                            result[result_key] = fit_and_run(avm)
                            if control.test:
                                return

    def search_rf(n_months_back):
        'search over RandomForestRegressor HPs, appending to result'
        for n_estimators in n_estimators_seq:
            for max_features in max_features_seq:
                for max_depth in max_depth_seq:
                    print (
                        '%6d %3s %1d %4d %4s %3d' %
                        (control.arg.yyyymm, 'rfr', n_months_back,
                         n_estimators, max_features_s(max_features), max_depth)
                    )
                    avm = AVM.AVM(
                        model_name='RandomForestRegressor',
                        forecast_time_period=control.arg.yyyymm,
                        random_state=control.random_seed,
                        n_months_back=n_months_back,
                        n_estimators=n_estimators,  # number of boosting stages
                        max_depth=max_depth,  # max depth of any tree
                        max_features=max_features,  # how many features to test when splitting
                    )
                    result_key = ResultKeyRfr(
                        n_months_back,
                        n_estimators,
                        max_features,
                        max_depth,
                    )
                    result[result_key] = fit_and_run(avm)
                    if control.test:
                        return

    # grid search for all model types
    for n_months_back in n_months_back_seq:
        search_en(n_months_back)
        search_gbr(n_months_back)
        search_rf(n_months_back)
        if control.test:
            break

    return result


def main(argv):
    control = make_control(argv)
    if False:
        # avoid error in sklearn that requires flush to have no arguments
        sys.stdout = Logger(base_name=control.arg.base_name)
    print control

    samples = pd.read_csv(
        control.path_in,
        nrows=None if control.test else None,
    )
    print 'samples.shape', samples.shape

    result = do_val(control, samples)

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
        pdb.set_trace()
        pprint()
        pd.DataFrame()
        np.array()

    main(sys.argv)
