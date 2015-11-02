'''create charts showing results of cross validation to set randomforest HPs

INPUT FILES
 INPUT/[test-'ege-rfbound-YYYYMM-folds-NN.pickle

OUTPUT FILES
 WORKING/[test-]chart-02-VAR.data.pickle
 WORKING/[test-]chart-02-VAR-YYYY[-MM].pdf
where
 VAR in {max_depth | max_features}
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
cc = columns_contain


def usage(msg=None):
    print __doc__
    if msg is not None:
        print msg
    print 'usage  : python chart-01.py HP [--data] [--test]'
    print ' HP in {max_depth | max_features}'
    print ' --data: produce reduction of the input file, not the actual charts'
    print ' --test: run in test mode'
    sys.exit(1)


def make_control(argv):
    # return a Bunch

    print argv
    if len(argv) not in (1, 2, 3):
        usage('invalid number of arguments')

    if len(argv) == 1:
        usage()

    pcl = ParseCommandLine(argv)
    arg = Bunch(
        base_name='chart-02',
        hp=argv[1],
        data=pcl.has_arg('--data'),
        test=pcl.has_arg('--test'),
    )

    random_seed = 123
    random.seed(random_seed)

    dir_working = Path().dir_working()

    debug = False

    out_file_name_base = ('test-' if arg.test else '') + arg.base_name + '-' + arg.hp

    return Bunch(
        arg=arg,
        debug=debug,
        path_in_ege=dir_working + 'ege-rfbound-%s-*-folds-10.pickle' % arg.hp,
        path_out_base=dir_working + out_file_name_base,
        path_data=dir_working + arg.base_name + '-' + arg.hp + '.data.pickle',
        random_seed=random_seed,
        test=arg.test,
    )


def make_chart(df, control, ege_control):
    'write one txt file for each n_months_back'
    def make_subplot(test_period, n_months_back):
        'mutate the default axes'
        for i, n_estimators in enumerate(sorted(set(df.n_estimators))):
            mask = (
                (df.test_period == test_period) &
                (df.n_months_back == n_months_back) &
                (df.n_estimators == n_estimators)
            )
            subset = df.loc[mask]
            x = subset.max_depth
            y = subset.mean_loss
            plt.plot(y / 1000.0,
                     label=('n_estimators: %d' % n_estimators),
                     # linestyle='.,ov^<>'[i],
                     linestyle=[':', '-.', '--', '-'][i % 4],
                     color='bgrcmykw'[i % 8],
                     )
        # plt.axis([0, len(y), 0, df.mean_loss.max() + 20000])
        plt.xticks(range(len(y)), x.values, size='xx-small', rotation='vertical')
        plt.yticks(size='xx-small')
        plt.title('yr-mo %s-%s bk %d' % (test_period[:4], test_period[4:], n_months_back),
                  loc='left',
                  fontdict={'fontsize': 'xx-small', 'style': 'italic'},
                  )
        # plt.legend(loc='best')
        return

    def make_figure(year, months):
        print 'make_figure', year, months
        test_periods_typical = [str(year * 100 + month)
                                for month in months
                                ]
        test_periods = ('200902',) if year == 2009 else test_periods_typical

        plt.figure()  # new figure
        # plt.suptitle('Loss by Test Period, Tree Max Depth, N Trees')  # overlays the subplots
        axes_number = 0
        n_months_backs = range(1, 7, 1)
        last_test_period_index = len(test_periods) - 1
        last_n_months_back_index = len(n_months_backs) - 1
        for test_period_index, test_period in enumerate(test_periods):
            for n_months_back_index, n_months_back in enumerate(n_months_backs):
                axes_number += 1  # count across rows
                plt.subplot(len(test_periods), len(n_months_backs), axes_number)
                make_subplot(test_period, n_months_back)
                if test_period_index == last_test_period_index:
                    # annotate the bottom row only
                    if n_months_back_index == 0:
                        plt.xlabel('max_depth')
                        plt.ylabel('loss x $1000')
                    if n_months_back_index == last_n_months_back_index:
                        plt.legend(loc='best', fontsize=5)
                        # plt.legend(loc='best', fontsize='xx-small')

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        out_suffix = '-%02d' % months if len(months) == 1 else ''
        plt.savefig(control.path_out_base + '-' + str(year) + out_suffix + '.pdf')
        plt.close()

    for year in (2004, 2005, 2006, 2007, 2008, 2009):
        months = (2,) if year == 2009 else (2, 5, 8, 11)
        for month in months:
            make_figure(year, (month,))
        make_figure(year, months)
        if control.test:
            break


def make_data(control):
    'return data frame, ege_control'
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
        test_period = path.split('.')[2].split('/')[3].split('-')[3]
        print 'reducing', path
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
                    'mean_loss': -grid_score.mean_validation_score,
                    'std_loss': np.std(grid_score.cv_validation_scores),
                }
            )
        if verbose:
            print 'number of grid search cells', len(gscv.grid_scores_)
            print 'best score', gscv.best_score_
            print 'best estimator', gscv.best_estimator_
            print 'best params'
            print_params(gscv.best_params_)
            print 'scorer', gscv.scorer_
        return ege_control

    rows_list = []
    for file in glob.glob(control.path_in_ege):
        ege_control = process_file(file, rows_list)
    df = pd.DataFrame(rows_list)
    return df, ege_control  # return last ege_control, not all


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(base_name=control.arg.base_name)
    print control

    if control.arg.data:
        df, ege_control = make_data(control)
        with open(control.path_data, 'wb') as f:
            pickle.dump((df, ege_control, control), f)
    else:
        with open(control.path_data, 'rb') as f:
            df, ege_control, data_control = pickle.load(f)
        make_chart(df, control, ege_control)

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
