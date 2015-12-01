'''Determine accuracy on validation set YYYYMM of various hyperparameter setting
for gradient boosted regression trees

INVOCATION
  python gbrtval.py YYYYMM [-test]

INPUT FILE:
  WORKING/samples-train-validate.csv

OUTPUT FILE:
  WORKING/gbrtval/YYYYMM.pickle
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
        base_name='gbrtval',
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

    return Bunch(
        arg=arg,
        debug=debug,
        path_in=dir_working + 'samples-train-validate.csv',
        path_out=dir_path + out_file_name,
        random_seed=random_seed,
        test=arg.test,
    )

ResultKey = collections.namedtuple('ResultKey',
                                   'n_months_back loss alpha max_depth max_features yyyymm',
                                   )
ResultValue = collections.namedtuple('ResultValue',
                                     'actuals predictions',
                                     )


def do_gbrtval(control, samples):
    'run grid search on elastic net and random forest models'

    # HP settings to test
    # common across models
    n_months_back_seq = (1, 2, 3, 4, 5, 6)
    # for GBRT
    loss_alphas = (('ls', None),        # loss function and corresponding alpha value
                   ('lad', None),
                   ('huber', 0.5),
                   ('quantile', 0.9),
                   ('quantile', 0.5),
                   ('quantile', 0.9),
                   )
    max_depths = (1, 2, 3)
    max_featuress = ('auto', 'sqrt', 'log2', None)

    result = {}

    def run(n_months_back, loss, alpha, max_depth, max_features):
        avm = AVM.AVM(
            model_name='GradientBoostingRegressor',
            forecast_time_period=control.arg.yyyymm,
            n_months_back=n_months_back,
            random_state=control.random_seed,
            loss=loss,
            alpha=alpha,
            max_depth=max_depth,
            max_features=max_features,
            verbose=0,
        )
        avm.fit(samples)
        mask = samples[layout_transactions.yyyymm] == control.arg.yyyymm
        samples_yyyymm = samples[mask]
        predictions = avm.predict(samples_yyyymm)
        actuals = samples_yyyymm[layout_transactions.price]
        result_key = ResultKey(n_months_back, loss, alpha, max_depth, max_features, control.arg.yyyymm)
        result[result_key] = ResultValue(actuals, predictions)

    pdb.set_trace()
    for n_months_back in n_months_back_seq:
        for loss_alpha in loss_alphas:
            loss, alpha = loss_alpha
            for max_depth in max_depths:
                for max_features in max_featuress:
                    print '%d %02d %8s %4s %d %4s' % (
                        control.arg.yyyymm, n_months_back, loss, alpha, max_depth, max_features)
                    run(n_months_back, loss, alpha, max_depth, max_features)

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

    result = do_gbrtval(control, samples)

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
