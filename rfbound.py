'''program to estimate the generalization error from a variety of AVMs

INPUT FILE:
    WORKING/samples-train-validate.csv
OUTPUT FILE:
    WORKING/rfbound/[test-]HP-YYYYMM-NN.pickle
'''

from __future__ import division

import cPickle as pickle
import numpy as np
import os
import pandas as pd
import pdb
from pprint import pprint
import random
import sklearn
import sklearn.grid_search
import sklearn.metrics
import sys

import AVM
from Bunch import Bunch
from columns_contain import columns_contain
import layout_transactions as transactions
from Logger import Logger
from ParseCommandLine import ParseCommandLine
from Path import Path
# from TimeSeriesCV import TimeSeriesCV
cc = columns_contain


def usage(msg=None):
    print __doc__
    if msg is not None:
        print msg
    print 'usage : python rfbound.py HP YYYYMM NN [--test]'
    print ' HP      {max_depth | max_features}'
    print ' YYYYMM  year + month; ex: 200402'
    print ' NN      number of folds to use for the cross validating'
    print ' --test  run in test mode (on a small sample of the entire data)',
    sys.exit(1)


def make_control(argv):
    # return a Bunch

    print argv
    if not(4 <= len(argv) <= 5):
        usage('invalid number of arguments')

    pcl = ParseCommandLine(argv)
    arg = Bunch(
        base_name='rfbound',
        hp=argv[1],
        yyyymm=argv[2],
        folds=argv[3],
        test=pcl.has_arg('--test'),
    )

    try:
        arg.folds = int(arg.folds)
    except:
        usage('INT not an integer; ' + str(arg.folds))

    random_seed = 123
    random.seed(random_seed)

    dir_working = Path().dir_working()

    debug = False

    out_file_name = (
        '%s/%s%s-%s-folds-%02d.pickle' % (
            arg.base_name,
            ('test-' if arg.test else ''),
            arg.hp,
            arg.yyyymm,
            arg.folds)
    )

    # assure the output directory exists
    dir_path = dir_working + arg.base_name
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return Bunch(
        arg=arg,
        debug=debug,
        path_in=dir_working + 'samples-train-validate.csv',
        path_out=dir_working + out_file_name,
        random_seed=random_seed,
        test=arg.test,
    )


def print_gscv(gscv, tag=None, only_best=False):
    pdb.set_trace()
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


def do_rfbound(control, samples):
    'run grid search on random forest model; return grid search object'

    # HP settings to test
    # common across --rfbound options
    model_name_seq = ('RandomForestRegressor',)
    n_months_back_seq = (1, 2, 3, 4, 5, 6)
    n_estimators_seq = (10, 30, 100, 300, 1000)
    # not common across --rfbound options
    max_features_seq = (1, 'log2', 'sqrt', .1, .3, 'auto')
    max_depth_seq = (1, 3, 10, 30, 100, 300)

    gscv = sklearn.grid_search.GridSearchCV(
        estimator=AVM.AVM(),
        param_grid=dict(
            model_name=model_name_seq,
            n_months_back=n_months_back_seq,
            forecast_time_period=[int(control.arg.yyyymm)],
            n_estimators=n_estimators_seq,
            max_depth=max_depth_seq if control.arg.hp == 'max_depth' else [None],
            max_features=max_features_seq if control.arg.hp == 'max_features' else [None],
            random_state=[control.random_seed],
        ),
        scoring=AVM.avm_scoring,
        n_jobs=1 if control.test else -1,
        cv=control.arg.folds,
        verbose=1 if control.test else 0,
    )
    gscv.fit(samples)
    print 'gscv'
    pprint(gscv)
    # print_gscv(gscv, tag=control.arg.rfbound, only_best=True)
    return gscv


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

    result = do_rfbound(control, samples)

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
        print transactions

    main(sys.argv)
