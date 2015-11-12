'''program to estimate the generalization error from a variety of AVMs

Determine accuracy on validation set YYYYMM of various hyperparameter setting
for a random forests model.

INPUT FILE:
    WORKING/samples-train-validate.csv
OUTPUT FILE:
    WORKING/rfval-YYYYMM.pickle
'''

from __future__ import division

import collections
import cPickle as pickle
import numpy as np
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
    print 'usage  : python rfval.py YYYYMM [--test]'
    print ' HP  {max_depth | max_features}'
    print ' YYYYMM  year + month; ex: 200402'
    print ' INT     number of folds to use for the cross validating'
    print ' --test      : run in test mode (on a small sample of the entire data)',
    sys.exit(1)


def make_control(argv):
    # return a Bunch

    print argv
    if not(2 <= len(argv) <= 3):
        usage('invalid number of arguments')

    pcl = ParseCommandLine(argv)
    arg = Bunch(
        base_name='rfval',
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
        '%s-%s.pickle' % (arg.base_name, arg.yyyymm)
    )

    return Bunch(
        arg=arg,
        debug=debug,
        path_in=dir_working + 'samples-train-validate.csv',
        path_out=dir_working + out_file_name,
        random_seed=random_seed,
        test=arg.test,
    )

ResultKey = collections.namedtuple('ResultKey',
                                   'n_months_back n_estimators max_depth max_features hp yyyymm',
                                   )
ResultValue = collections.namedtuple('ResultValue',
                                     'actuals predictions rmse',
                                     )


def do_rfval(control, samples):
    'run grid search on random forest model; return grid search object'

    # HP settings to test
    # common across --rfbound options
    n_months_back_seq = (1, 2, 3, 4, 5, 6)
    n_estimators_seq = (10, 30, 100, 300, 1000)
    hp_seq = ('max_depth', 'max_features')
    # not common across --rfbound options
    max_features_seq = (1, 'log2', 'sqrt', .1, .3, 'auto')
    max_depth_seq = (1, 3, 10, 30, 100, 300)

    result = {}

    def run(n_months_back, n_estimators, max_depth, max_features):
        assert (max_depth is not None) or (max_features is not None)
        avm = AVM.AVM(
            model_name='RandomForestRegressor',
            forecast_time_period=control.arg.yyyymm,
            n_months_back=n_months_back,
            random_state=control.random_seed,
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
        )
        avm.fit(samples)
        mask = samples[layout_transactions.yyyymm] == control.arg.yyyymm
        samples_yyyymm = samples[mask]
        predictions = avm.predict(samples_yyyymm)
        actuals = samples_yyyymm[layout_transactions.price]
        errors = actuals - predictions
        mse = np.sum(errors * errors) / len(actuals)
        rmse = np.sqrt(mse)
        result_key = ResultKey(n_months_back, n_estimators, max_depth, max_features, hp,
                               control.arg.yyyymm)
        print result_key, rmse
        result[result_key] = ResultValue(actuals, predictions, rmse)

    for n_months_back in n_months_back_seq:
        for n_estimators in n_estimators_seq:
            for hp in hp_seq:
                if hp == 'max_depth':
                    max_features = None
                    for max_depth in max_depth_seq:
                        run(n_months_back, n_estimators, max_depth, max_features)
                elif hp == 'max_features':
                    max_depth = None
                    for max_features in max_features_seq:
                        run(n_months_back, n_estimators, max_depth, max_features)
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

    result = do_rfval(control, samples)

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
