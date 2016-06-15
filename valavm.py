'''Determine accuracy on validation set YYYYMM of various hyperparameter setting
for AVMs based on 3 models (linear, random forests, gradient boosting regression)

INVOCATION
  python valavm.py TEST_MONTH [--test] [--onlyfitted K]
  where
   TEST_MONTH: yyyymm  Month of test data; training uses months just prior
   K: int              Number of best-fitting models to use in grid

INPUTS
 WORKING/samples-train.csv
 WORKING/rank_models/TEST_MONTH-fitted.pickle  HPs for best models

OUTPUTS
 WORKING/valavm/TEST_MONTH.pickle         contains result from grid search
 WORKING/valavm/TEST_MONTH-fitted.pickle  containing K best fitting
'''

from __future__ import division

import argparse
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
# from Features import Features
import layout_transactions
from Logger import Logger
from Month import Month
from Path import Path
from SampleSelector import SampleSelector
from Timer import Timer
# from TimeSeriesCV import TimeSeriesCV
cc = columns_contain


def make_grid():
    # return Bunch of hyperparameter settings
    return Bunch(
        # HP settings to test across all models
        n_months_back_seq=(1, 2, 3, 6, 12),

        # HP settings to test for ElasticNet models
        alpha_seq=(0.01, 0.03, 0.1, 0.3, 1.0),  # multiplies the penalty term
        l1_ratio_seq=(0.0, 0.25, 0.50, 0.75, 1.0),  # 0 ==> L2 penalty, 1 ==> L1 penalty
        units_X_seq=('natural', 'log'),
        units_y_seq=('natural', 'log'),

        # HP settings to test for tree-based models
        # settings based on Anil Kocak's recommendations
        n_estimators_seq=(10, 30, 100, 300),
        max_features_seq=(1, 'log2', 'sqrt', 'auto'),
        max_depth_seq=(1, 3, 10, 30, 100, 300),

        # HP setting to test for GradientBoostingRegression models
        learning_rate_seq=(.10, .25, .50, .75, .99),
        # experiments demonstrated that the best loss is seldom quantile
        # loss_seq=('ls', 'quantile'),
        loss_seq=('ls',),
    )


def positive_int(s):
    try:
        value = int(s)
        assert value > 0
        return value
    except:
        raise argparse.ArgumentTypeError('%s is not a positive integer' % s)


def year_month(s):
    'return s, only if it is form YYYYMM'
    s_year = s[:4]
    s_month = s[4:]
    try:
        int_year = int(s_year)
        assert 0 <= int_year <= 2016

        int_month = int(s_month)
        assert 1 <= int_month <= 12

        return s
    except:
        raise argparse.ArgumentTypeError('%s is not a yearmonth' % s)


def make_control(argv):
    'return a Bunch'
    print argv
    parser = argparse.ArgumentParser()
    parser.add_argument('invocation')
    parser.add_argument('test_month', type=year_month)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--onlyfitted', action='store', type=positive_int)
    arg = parser.parse_args(argv)
    arg.base_name = 'valavm'

    random_seed = 123
    random.seed(random_seed)

    dir_working = Path().dir_working()

    # assure output directory exists
    dir_path = dir_working + arg.base_name + '/'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return Bunch(
        arg=arg,
        debug=False,
        path_in_samples=dir_working + 'samples-train.csv',
        path_in_ranked_models_dir=dir_working + 'rank_models/',
        path_out_file=dir_path + arg.test_month + '.pickle',
        path_out_fitted=dir_path + 'fitted-' + arg.test_month + '.pickle',
        path_out_log=dir_path + 'log-' + str(arg.test_month) + '.txt',
        grid=make_grid(),
        random_seed=random_seed,
        timer=Timer(),
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


def do_val(control, samples, save, already_exists):
    'run grid search on control.grid.hyperparameters across the 3 model kinds'

    def check_for_missing_predictions(result):
        for k, v in result.iteritems():
            if v.predictions is None:
                print k
                print 'found missing predictions'
                pdb.set_trace()

    def max_features_s(max_features):
        'convert to 4-character string (for printing)'
        return max_features[:4] if isinstance(max_features, str) else ('%4.1f' % max_features)

    result = {}

    def fit_and_run(avm, samples_test, samples_train):
        'return a ResultValue'
        pdb.set_trace()
        avm.fit(samples_train)
        predictions = avm.predict(samples_test)
        if predictions is None:
            print 'no predictions!'
            pdb.set_trace()
        actuals = samples_test[layout_transactions.price]
        return ResultValue(actuals, predictions)

    def search_en(samples_test, samples_train):
        'search over ElasticNet HPs, appending to result'
        for units_X in control.grid.units_X_seq:
            for units_y in control.grid.units_y_seq:
                for alpha in control.grid.alpha_seq:
                    for l1_ratio in control.grid.l1_ratio_seq:
                        print (
                            control.arg.test_month, 'en', n_months_back, units_X[:3], units_y[:3],
                            alpha, l1_ratio,
                        )
                        avm = AVM.AVM(
                            model_name='ElasticNet',
                            random_state=control.random_seed,
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
                        if already_exists(result_key):
                            print 'already exists'
                        else:
                            print
                            save(result_key,
                                 fit_and_run(avm, samples_test, samples_train))
                            if control.arg.test:
                                return

    def search_gbr(samples_test, samples_train):
        'search over GradientBoostingRegressor HPs, appending to result'
        for n_estimators in control.grid.n_estimators_seq:
            for max_features in control.grid.max_features_seq:
                for max_depth in control.grid.max_depth_seq:
                    for loss in control.grid.loss_seq:
                        for learning_rate in control.grid.learning_rate_seq:
                            print (
                                control.arg.test_month, 'gbr', n_months_back,
                                n_estimators, max_features_s(max_features), max_depth, loss, learning_rate,
                            )
                            avm = AVM.AVM(
                                model_name='GradientBoostingRegressor',
                                random_state=control.random_seed,
                                learning_rate=learning_rate,
                                loss=loss,
                                alpha=.5 if loss == 'quantile' else None,
                                n_estimators=n_estimators,  # number of boosting stages
                                max_depth=max_depth,  # max depth of any tree
                                max_features=max_features,  # how many features to test whenBoosting splitting
                            )
                            result_key = ResultKeyGbr(
                                n_months_back,
                                n_estimators,
                                max_features,
                                max_depth,
                                loss,
                                learning_rate,
                            )
                            if already_exists(result_key):
                                print 'already exists'
                            else:
                                print
                                save(result_key,
                                     fit_and_run(avm, samples_test, samples_train))
                                if control.arg.test:
                                    return

    def search_rf(samples_test, samples_train):
        'search over RandomForestRegressor HPs, appending to result'
        for n_estimators in control.grid.n_estimators_seq:
            for max_features in control.grid.max_features_seq:
                for max_depth in control.grid.max_depth_seq:
                    print (
                        control.arg.test_month, 'rfr', n_months_back,
                        n_estimators, max_features_s(max_features), max_depth
                    )
                    avm = AVM.AVM(
                        model_name='RandomForestRegressor',
                        random_state=control.random_seed,
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
                    if already_exists(result_key):
                        print 'already exists'
                    else:
                        print
                        save(result_key,
                             fit_and_run(avm, samples_test, samples_train))
                        if control.arg.test:
                            return

    # grid search for all model types
    for n_months_back in control.grid.n_months_back_seq:
        print n_months_back
        test_month = Month(control.arg.test_month)
        ss = SampleSelector(samples)
        samples_test = ss.in_month(test_month)
        samples_train = ss.between_months(
            test_month.decrement(n_months_back),
            test_month.decrement(1),
        )
        print n_months_back, test_month.decrement(n_months_back), test_month.decrement(1)
        print len(samples_test), len(samples_train)
        pdb.set_trace()
        search_en(samples_test, samples_train)
        search_gbr(samples_test, samples_train)
        search_rf(samples_test, samples_train)

    return result


FittedAvm = collections.namedtuple('FittedAVM', 'index key fitted')


def process_only_fitted(control, samples):
    'write files containing fitted models and feature names'
    # WRITEME: duplicate some functionality in do_val
    # get feature names from Features().ege_names(), which returns a tuple of strings
    def fit_avm(key):
        'return attributes of fitted model'
        avm = AVM.AVM(
            model_name={
                'en': 'ElasticNet',
                'gb': 'GradientBoostingRegressor',
                'rf': 'RandomForestRegressor',
                }[key.model],
            forecast_time_period=control.arg.test_month,
            n_months_back=key.n_months_back,
            random_state=control.random_seed,
            verbose=True,
            alpha=key.alpha,
            l1_ratio=key.l1_ratio,
            units_X=key.units_X,
            units_y=key.units_y,
            n_estimators=key.n_estimators,
            max_depth=key.max_depth,
            max_features=key.max_features,
            learning_rate=key.learning_rate,
            loss=key.loss,
            )
        avm.fit(samples)
        return avm.model  # caller wants the attributes of the fitted model

    def fit_and_write_k_best(best_models_f):
        with open(control.path_out_fitted, 'wb') as g:
            try:
                pickled = pickle.load(best_models_f)  # read until EOF or until K records are processed
                for index, key in enumerate(pickled):  # process the SortedDictionary
                    if index >= control.arg.onlyfitted:
                        # process only K best
                        break
                    print 'fitting index', index, 'of', control.arg.onlyfitted
                    print 'fitting key', key
                    fitted_avm = fit_avm(key)
                    # drop all the fitted sub-estimators, because they take a lot of room
                    if key.model in ('gb', 'rf'):
                        fitted_avm.estimators_ = None
                    print 'appending fitted avm to', control.path_out_fitted
                    pickle.dump((index, key, fitted_avm), g)
                    # pickle.dump(fa, g)
            except EOFError:
                print 'found EOF for test_month:', control.arg.test_month

    path = control.path_in_ranked_models_dir + control.arg.test_month + '.pickle'
    print 'reading ranked model descriptions from', path
    with open(path, 'rb') as best_models_f:
        fit_and_write_k_best(best_models_f)


def process_complete_grid(control, samples):
    pdb.set_trace()
    existing_keys = set()
    with open(control.path_out_file, 'rb') as prior:
        while True:
            try:
                record = pickle.load(prior)
                key, value = record
                existing_keys.add(key)
            except ValueError as e:
                print record
                print e
                print 'ignored'
            except EOFError:
                break
    pdb.set_trace()
    print 'number of existing keys in output file:', len(existing_keys)

    with open(control.path_out_file, 'ab') as output:
        def already_exists(key):
            return key in existing_keys

        def save(key, value):
            print 'save key=', key
            record = (key, value)
            pickle.dump(record, output)

        do_val(control, samples, save, already_exists)


def main(argv):
    control = make_control(argv)
    if False:
        # avoid error in sklearn that requires flush to have no arguments
        sys.stdout = Logger(log_file_path=control.path_out_log)
    print control

    samples = pd.read_csv(
        control.path_in_samples,
        nrows=None if control.arg.test else None,
    )
    print 'samples.shape', samples.shape
    control.timer.lap('read samples')

    # assure output file exists
    if not os.path.exists(control.path_out_file):
        os.system('touch %s' % control.path_out_file)

    if control.arg.onlyfitted is None:
        process_complete_grid(control, samples)
    else:
        process_only_fitted(control, samples)

#    existing_keys = set()
#    with open(control.path_out_file, 'rb') as prior:
#        while True:
#            try:
#                record = pickle.load(prior)
#                key, value = record
#                existing_keys.add(key)
#            except ValueError as e:
#                print record
#                print e
#                print 'ignored'
#            except EOFError:
#                break
#    print 'number of existing keys in output file:', len(existing_keys)
#
#    with open(control.path_out_file, 'ab') as output:
#        def already_exists(key):
#            return key in existing_keys
#
#        def save(key, value):
#            print 'save key=', key
#            record = (key, value)
#            pickle.dump(record, output)
#
#        do_val(control, samples, save, already_exists)

    print control
    if control.arg.test:
        print 'DISCARD OUTPUT: test'
    if control.debug:
        print 'DISCARD OUTPUT: debug'
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
