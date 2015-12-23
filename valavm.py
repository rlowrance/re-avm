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

ResultKey = collections.namedtuple('ResultKey',
                                   'n_months_back learning_rate yyyymm',
                                   )
ResultValue = collections.namedtuple('ResultValue',
                                     'actuals predictions',
                                     )


def do_val(control, samples):
    'run grid search on elastic net and random forest models'

    def check_for_missing_predictions(result):
        for k, v in result.iteritems():
            if v.predictions is None:
                print k
                print 'found missing predictions'
                pdb.set_trace()

    # HP settings to test
    n_months_back_seq = (1, 2, 3, 4, 5, 6)
    learning_rate_seq = (.10, .20, .30, .40, .50, .60, .70, .80, .90)

    result = {}

    def run(n_months_back, learning_rate):
        # fix loss as quantile .50
        # max_depth: use default
        # max_features: use default

        print (
            'gbrval %6d %1d %5.3f' %
            (control.arg.yyyymm, n_months_back, learning_rate)
        )
        avm = AVM.AVM(
            model_name='GradientBoostingRegressor',
            forecast_time_period=control.arg.yyyymm,
            n_months_back=n_months_back,
            random_state=control.random_seed,
            loss=control.fixed_hps.loss,
            alpha=control.fixed_hps.alpha,
            learning_rate=learning_rate,
            n_estimators=control.fixed_hps.n_estimators,
            max_depth=control.fixed_hps.max_depth,
            max_features=control.fixed_hps.max_features,
            verbose=0,
        )
        avm.fit(samples)
        mask = samples[layout_transactions.yyyymm] == control.arg.yyyymm
        samples_yyyymm = samples[mask]
        predictions = avm.predict(samples_yyyymm)
        if predictions is None:
            pdb.set_trace()
        actuals = samples_yyyymm[layout_transactions.price]
        result_key = ResultKey(n_months_back, learning_rate, control.arg.yyyymm)
        result[result_key] = ResultValue(actuals, predictions)

    for n_months_back in n_months_back_seq:
        for learning_rate in learning_rate_seq:
            run(n_months_back, learning_rate)
        if control.test:
            break

    check_for_missing_predictions(result)
    return result


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
