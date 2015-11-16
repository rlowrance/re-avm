'''program to estimate the generalization error from a variety of AVMs

Determine accuracy on validation set YYYYMM of various hyperparameter setting
for elastic net.

INVOCATION
  python val.py YYYYMM [-test]

INPUT FILE:
  WORKING/samples-train-validate.csv

OUTPUT FILE:
  WORKING/linval/YYYYMM.pickle
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
        base_name='linval',
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
                                   'n_months_back alpha l1_ratio units_X units_y yyyymm',
                                   )
ResultValue = collections.namedtuple('ResultValue',
                                     'actuals predictions',
                                     )


def do_linval(control, samples):
    'run grid search on elastic net and random forest models'

    # HP settings to test
    # common across models
    n_months_back_seq = (1, 2, 3, 4, 5, 6)
    # for ElasticNet
    # TODO: decide what to do about alphe == 0
    alpha_seq = (0.1, 0.3, 1.0, 3.0)  # multiplies the penalty term
    l1_ratio_seq = (0.0, 0.25, 0.50, 0.75, 1.0)  # 0 ==> L2 penalty, 1 ==> L1 penalty
    units_X_seq = ('natural', 'log')
    units_y_seq = ('natural', 'log')

    result = {}

    def run(n_months_back, alpha, l1_ratio, units_X, units_y):
        avm = AVM.AVM(
            model_name='ElasticNet',
            forecast_time_period=control.arg.yyyymm,
            n_months_back=n_months_back,
            random_state=control.random_seed,
            alpha=alpha,
            l1_ratio=l1_ratio,
            units_X=units_X,
            units_y=units_y,
            verbose=0,
        )
        avm.fit(samples)
        mask = samples[layout_transactions.yyyymm] == control.arg.yyyymm
        samples_yyyymm = samples[mask]
        predictions = avm.predict(samples_yyyymm)
        actuals = samples_yyyymm[layout_transactions.price]
        result_key = ResultKey(n_months_back, alpha, l1_ratio, units_X, units_y, control.arg.yyyymm)
        result[result_key] = ResultValue(actuals, predictions)

    for n_months_back in n_months_back_seq:
        for alpha in alpha_seq:
            for l1_ratio in l1_ratio_seq:
                for units_X in units_X_seq:
                    for units_y in units_y_seq:
                        print '%d %02d %4.2f %4.2f %7s %7s' % (
                            control.arg.yyyymm, n_months_back, alpha, l1_ratio, units_X, units_y)
                        run(n_months_back, alpha, l1_ratio, units_X, units_y)
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

    result = do_linval(control, samples)

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
