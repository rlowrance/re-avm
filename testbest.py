'''Test the best models found for each month in 2007

INVOCATION
  python testbest.py [--test]

INPUT FILES:
  WORKING/samples-train.csv             training data
  WORKING/chart-06/best.pickle          describes the best models found by valavm and chart-06

OUTPUT FILE:
  WORKING/testbest/results.pickle       run chart-07.py to create charts using these data
'''

from __future__ import division

import cPickle as pickle
import numpy as np
import os
import pandas as pd
import pdb
from pprint import pprint
import random
import sys

import AVM2 as AVM
from Bunch import Bunch
from columns_contain import columns_contain
import layout_transactions
from Logger import Logger
from make_test_train import make_test_train
from Month import Month
from ParseCommandLine import ParseCommandLine
from Path import Path
from Timer import Timer
import valavm
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
    if not(1 <= len(argv) <= 2):
        usage('invalid number of arguments')

    pcl = ParseCommandLine(argv)
    arg = Bunch(
        base_name='testbest',
        test=pcl.has_arg('--test'),
    )

    random_seed = 123
    random.seed(random_seed)

    dir_working = Path().dir_working()

    debug = False

    out_file_name = (
        ('test-' if arg.test else '') +
        '%s.pickle' % 'results'
    )

    # assure output directory exists
    dir_path = dir_working + arg.base_name + '/'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return Bunch(
        arg=arg,
        debug=debug,
        path_in_data=dir_working + 'samples-train.csv',
        path_in_best=dir_working + 'chart-06/best.pickle',
        path_out=dir_path + out_file_name,
        random_seed=random_seed,
        test=arg.test,
    )


def make_model_grid(value):
    'suffix element names with "_seq"'
    s = value[0]
    print s
    print type(s)
    is_en = s.model == 'en'
    is_gb = s.model == 'gb'
    is_tree = (s.model == 'gb') or (s.model == 'rf')
    return (
        s.model,
        Bunch(
            alpha_seq=(s.alpha,) if is_tree else None,
            l1_ratio_seq=(s.l1_ratio,) if is_tree else None,
            learning_rate_seq=(s.learning_rate,) if is_gb else None,
            loss_seq=(s.loss,) if is_gb else None,
            max_depth_seq=(s.max_depth,) if is_tree else None,
            max_features_seq=(s.max_features,) if is_tree else None,
            n_estimators_seq=(s.n_estimators,) if is_tree else None,
            n_months_back_seq=(s.n_months_back,),
            units_X_seq=(s.units_X,) if is_en else None,
            units_y_seq=(s.units_y,) if is_en else None,
        ))


def make_avm(forecast_time_period, random_seed, series):
    'return AVM instance'

    # the constructor calls mimic the constructor codes in valavm.py
    def en():
        return AVM.AVM(
            model_name='ElasticNet',
            forecast_time_period=forecast_time_period,
            random_state=random_seed,
            n_months_back=series.n_months_back,
            units_X=series.units_X,
            units_y=series.units_y,
            alpha=series.alpha,
            l1_ratio=series.l1_ratio,
        )

    def gb():
        return AVM.AVM(
            model_name='GradientBoostingRegressor',
            forecast_time_period=forecast_time_period,
            random_state=random_seed,
            n_months_back=series.n_months_back,
            learning_rate=series.learning_rate,
            loss=series.loss,
            alpha=.5 if series.loss == 'quantile' else None,
            n_estimators=series.n_estimators,
            max_depth=series.max_depth,
            max_features=series.max_features,
        )

    def rf():
        return AVM.AVM(
            model_name='RandomForestRegressor',
            forecast_time_period=forecast_time_period,
            random_state=random_seed,
            n_months_back=series.n_months_back,
            n_estimators=series.n_estimators,
            max_depth=series.max_depth,
            max_features=series.max_features,
        )

    # convert certain floating point values to ints
    # in order to make GradientBoostingRegressor happy

    series.max_depth = int(series.max_depth)
    series.n_estimators = int(series.n_estimators)
    series.n_months_back = int(series.n_months_back)

    if series.model == 'en':
        return en()
    elif series.model == 'gb':
        return gb()
    elif series.model == 'rf':
        return rf()
    else:
        print 'bad series.model'
        print series.model
        pdb.set_trace()


def do_testbest(control, samples, best):
    'determine accuracy of the HPs sets found to be best by valavm (as reported by chart-06)'
    result = {}
    for test_period, value in best.iteritems():
        forecast_month = Month(test_period).increment()
        series = value[0]
        print test_period, forecast_month
        print series
        avm = make_avm(forecast_month, control.random_seed, series)
        test_df, train_df = make_test_train(
            forecast_month,
            series.n_months_back,
            layout_transactions.yyyymm,
            samples,
        )
        avm.fit(train_df)  # the AVM object knows how to extract the train and test samples
        predictions = avm.predict(test_df)
        actuals = samples[layout_transactions.price]
        result[forecast_month] = valavm.ResultValue(actuals, predictions)
    return result


def main(argv):
    timer = Timer()
    control = make_control(argv)
    if False:
        # avoid error in sklearn that requires flush to have no arguments
        sys.stdout = Logger(base_name=control.arg.base_name)
    print control

    samples = pd.read_csv(
        control.path_in_data,
        nrows=None if control.test else None,
    )
    print 'samples.shape', samples.shape

    with open(control.path_in_best, 'rb') as f:
        best = pickle.load(f)

    result = do_testbest(control, samples, best)

    with open(control.path_out, 'wb') as f:
        pickle.dump((result, control), f)

    print 'elapsed wall clock seconds:', timer.elapsed_wallclock_seconds()
    print 'elapsed CPU seconds       :', timer.elapsed_cpu_seconds()

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
