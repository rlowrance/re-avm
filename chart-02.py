'''create charts showing results of cross validation to set randomforest HPs

INPUT FILES
 INPUT/[test-'ege-rfbound-YYYYMM-folds-NN.pickle

OUTPUT FILES
 WORKING/chart-02-YYYYMM-n_months_back-NN.data.pickle
 WORKING/chart-02-YYYYMM.txt
'''

from __future__ import division

import cPickle as pickle
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb
from pprint import pprint
import random
import sys

from AVM import AVM
from Bunch import Bunch
from columns_contain import columns_contain
from Logger import Logger
from ParseCommandLine import ParseCommandLine
from Path import Path
from Report import Report
cc = columns_contain


def usage(msg=None):
    print __doc__
    if msg is not None:
        print msg
    print 'usage  : python chart-01.py [--data] [--test]'
    print ' --data: produce reduction of the input file, not the actual charts'
    print ' --test: run in test mode'
    sys.exit(1)


def make_control(argv):
    # return a Bunch

    print argv
    if len(argv) not in (1, 2, 3):
        usage('invalid number of arguments')

    pcl = ParseCommandLine(argv)
    arg = Bunch(
        base_name='chart-02',
        data=pcl.has_arg('--data'),
        test=pcl.has_arg('--test'),
    )

    random_seed = 123
    random.seed(random_seed)

    dir_working = Path().dir_working()

    debug = False

    out_file_name_base = ('test-' if arg.test else '') + arg.base_name

    return Bunch(
        arg=arg,
        debug=debug,
        path_in_ege=dir_working + 'ege-rfbound-*-folds-10.pickle',
        path_out_txt_base=dir_working + out_file_name_base,
        path_data=dir_working + out_file_name_base + '.data.pickle',
        random_seed=random_seed,
        test=arg.test,
    )


def make_chart(df, control, ege_control):
    'write one txt file for each n_months_back'
    format_header = '%12s %12s %12s'
    format_detail = '%12d %12d %12.0f'

    def make_mean_loss(n_months_back, n_estimators, max_depth):
        mask1 = df.n_months_back == n_months_back
        mask2 = df.n_estimators == n_estimators
        mask3 = df.max_depth == max_depth
        mask = mask1 & mask2 & mask3
        selected = df.loc[mask]
        if len(selected) != 1:
            pdb.set_trace()
        assert len(selected) == 1, len(selected)
        mean_loss = - selected.mean_validation_score
        return mean_loss

    def n_estimators(r, n_months_back):
        r.append(format_header % ('n_estimators', 'max_depth', 'mean_loss'))
        # rely on the fact that the grid search was a grid
        for n_estimators in sorted(set(df.n_estimators)):
            for max_depth in sorted(set(df.max_depth)):
                mean_loss = make_mean_loss(
                    n_months_back=n_months_back,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                )
                r.append(format_detail % (n_estimators, max_depth, mean_loss))

    def max_depth(r, n_months_back):
        r.append(format_header % ('max_depth', 'n_estimators', 'mean_loss'))
        # rely on the fact that the grid search was a grid
        for max_depth in sorted(set(df.max_depth)):
            for n_estimators in sorted(set(df.n_estimators)):
                mean_loss = make_mean_loss(
                    n_months_back=n_months_back,
                    max_depth=max_depth,
                    n_estimators=n_estimators,
                )
                r.append(format_detail % (max_depth, n_estimators, mean_loss))

    def make_plot(n_months_back):
        pdb.set_trace()
        for i, n_estimators in enumerate(sorted(set(df.n_estimators))):
            mask = (
                (df.n_months_back == n_months_back) &
                (df.n_estimators == n_estimators)
            )
            subset = df.loc[mask]
            x = subset.max_depth
            y = -subset.mean_validation_score  # loss
            # AM: the lines don't show up
            plt.plot(y,
                     label=('n_estimators: %d' % n_estimators),
                     linestyle=[':', '-.', '- -', '-'][i % 4],
                     color='bgrcmybw'[i % 8],
                     )
        pdb.set_trace()
        print df.max_depth.max()
        plt.axis([0, len(y), 0, -df.mean_validation_score.max() + 20000])
        plt.xticks(range(len(y)), x.values)
        plt.title('Loss by max_depth and n_estimators')
        plt.xlabel('max_depth')
        plt.ylabel('loss')
        plt.legend(loc="best")
        # plt.show()  NOTE: this blocks if not in interactive mode
        return
        # MAYBE: create an axes for each n_months_back (there will be 12 of them)

    def out_file_base(n_months_back):
        return (
            control.path_out_txt_base +
            ('-%d' % control.yyyymm) +
            ('-n_months_back-%02d' % n_months_back))

    for n_months_back in sorted(set(df.n_months_back)):
        r = Report()
        r.append('CHART 02')
        r.append('MEAN LOSS FROM %d-FOLD CROSS VALIDATION' % ege_control.n_cv_folds)
        r.append('NUMBER OF MONTHS OF TRAINING DATA: %d' % n_months_back)
        r.append('')
        n_estimators(r, n_months_back)
        r.append('')
        max_depth(r, n_months_back)
        pdb.set_trace()
        r.write(out_file_base(n_months_back) + '.txt')

        make_plot(n_months_back)  # value is return in module plt
        pdb.set_trace()
        plt.savefig(out_file_base(n_months_back) + '.pdf')
        break  # while debugging


def make_data(control):
    'return data frame with columns: n_estimators, max_depth, n_months_back, loss'
    # FIXME: read all the files [test-]ege-rbound-YYYYMM-folds-NN.pickle'
    def print_params(params):
        for k, v in params.iteritems():
            print ' parameter %15s: %s' % (k, v)

    def print_grid_score(gs):
        print ' mean: %.0f std: %0.f' % (gs.mean_validation_score, np.std(gs.cv_validation_scores))
        for cv_vs in gs.cv_validation_scores:
            print ' validation score: %0.6f' % cv_vs
        print_params(gs.parameters)

    def process_file(path, rows_list):
        'mutate rows_list to include gscv object info at path'
        verbose = False
        test_period = path.split('.')[2].split('/')[3].split('-')[2]
        print 'reducing test_period', test_period
        with open(path, 'rb') as f:
            gscv, ege_control = pickle.load(f)
        # for now, just navigate the gscv object and print it
        # the gscv object is the result of running sklearn GridSearchCV
        if verbose:
            print 'headings: index, max_depth, n_estimators, n_months_back, mean score, std scores'
        for i, grid_score in enumerate(gscv.grid_scores_):
            # a grid_score is an instance of _CVScoreTuple, which has these fields:
            # parameters, mean_validation_score, cv_validation_scores
            if verbose:
                print '%3d %4d %4d %2d %7.0f %6.0f' % (
                    i,
                    grid_score.parameters['max_depth'],
                    grid_score.parameters['n_estimators'],
                    grid_score.parameters['n_months_back'],
                    grid_score.mean_validation_score,
                    np.std(grid_score.cv_validation_scores),
                )
            rows_list.append(
                {
                    'test_period': test_period,
                    'max_depth': grid_score.parameters['max_depth'],
                    'n_estimators': grid_score.parameters['n_estimators'],
                    'n_months_back': grid_score.parameters['n_months_back'],
                    'loss': grid_score.mean_validation_score,
                    'std': np.std(grid_score.cv_validation_scores),
                }
            )
        if verbose:
            print 'number of grid search cells', len(gscv.grid_scores_)
            print 'best score', gscv.best_score_
            print 'best estimator', gscv.best_estimator_
            print 'best params'
            print_params(gscv.best_params_)
            print 'scorer', gscv.scorer_

    rows_list = []
    for file in glob.glob(control.path_in_ege):
        process_file(file, rows_list)
    pdb.set_trace()
    df = pd.DataFrame(rows_list)
    return df


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(base_name=control.arg.base_name)
    print control

    if control.arg.data:
        pdb.set_trace()
        df = make_data(control)
        with open(control.path_data, 'wb') as f:
            pickle.dump((df, control), f)
    else:
        with open(control.path_data, 'rb') as f:
            df, data_control = pickle.load(f)
        make_chart(df, control)

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
        AVM()

    main(sys.argv)
