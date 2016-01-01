'''create charts showing results of valgbr.py

INVOCATION
  python chart-06.py [--data | YYYYMM] [--test]

INPUT FILES
 INPUT/valavm/YYYYMM.pickle

OUTPUT FILES
 WORKING/chart-06/[test-]data.pickle
 WORKING/chart-06/[test-]YYYY-a.pdf    comparison of losses by model
'''

from __future__ import division

import collections
import cPickle as pickle
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
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
from valavm import ResultKeyEn, ResultKeyGbr, ResultKeyRfr, ResultValue
cc = columns_contain

Key = collections.namedtuple(  # hold all posssible keys that valavm may have generated
    'Key',
    'yyyymm ' +                                # period
    'n_months_back units_X units_y ' +         # all ResultKey* have these fields
    'alpha l1_ratio ' +                        # only ResultKeyEn has these fields
    'n_estimators max_features max_depth ' +   # ResultKeyRfr and ResultKeyGbr have these fields
    'loss learning_rate',                      # only ResultKeyGbr has these fields
)

Value = collections.namedtuple(
    'Value',
    'mae',  # median absolute error
)


def usage(msg=None):
    print __doc__
    if msg is not None:
        print msg
    sys.exit(1)


def make_control(argv):
    # return a Bunch

    print argv
    if len(argv) not in (1, 2):
        usage('invalid number of arguments')

    pcl = ParseCommandLine(argv)
    arg = Bunch(
        base_name='chart-06',
        data=pcl.has_arg('--data'),
        test=pcl.has_arg('--test'),
    )

    random_seed = 123
    random.seed(random_seed)

    dir_working = Path().dir_working()

    debug = False

    reduced_file_name = ('test-' if arg.test else '') + 'data.pickle'

    # assure output directory exists
    dir_path = dir_working + arg.base_name + '/'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return Bunch(
        arg=arg,
        debug=debug,
        path_in_ege=dir_working + 'valavm/*.pickle',
        path_reduction=dir_path + reduced_file_name,
        path_chart_base=dir_path,
        random_seed=random_seed,
        test=arg.test,
    )


def make_charts_v1(df, control, ege_control):
    'write one txt file for each n_months_back'
    def make_subplot(test_period, n_months_back, loss_metric):
        'mutate the default axes'
        alphas = sorted(set(df.alpha))
        l1_ratios = sorted(set(df.l1_ratio))
        n_ticks = len(alphas) * len(l1_ratios)
        line_index = -1
        for units_X in ('natural', 'log'):
            for units_y in ('natural', 'log'):
                line_index += 1
                x_label = []
                y = np.empty((n_ticks,), dtype=float)
                tick_index = -1
                for alpha in alphas:
                    for l1_ratio in l1_ratios:
                        tick_index += 1
                        mask = (
                            (df.test_period == test_period) &
                            (df.n_months_back == n_months_back) &
                            (df.units_X == units_X) &
                            (df.units_y == units_y) &
                            (df.alpha == alpha) &
                            (df.l1_ratio == l1_ratio)
                        )
                        subset = df.loc[mask]
                        assert len(subset) == 1, subset
                        row = dict(subset.iloc[0])
                        y[tick_index] = row[loss_metric]
                        x_label.append('%3.1f-%4.2f' % (row['alpha'], row['l1_ratio']))
                plt.plot(y / 1000.0,
                         label='%3s-%3s' % (units_X, units_y),
                         linestyle=[':', '-.', '--', '-'][line_index % 4],
                         color='bgrcmykw'[line_index % 8],
                         )
                plt.xticks(range(len(y)), x_label, size='xx-small', rotation='vertical')
                plt.yticks(size='xx-small')
        plt.title('yr-mo %s-%s bk %d' % (test_period[:4], test_period[4:], n_months_back),
                  loc='left',
                  fontdict={'fontsize': 'xx-small', 'style': 'italic'},
                  )

    def make_figure(year, months):

        print 'make_figure', year, months
        test_periods_typical = [str(year * 100 + month)
                                for month in months
                                ]
        test_periods = ('200902',) if year == 2009 else test_periods_typical

        plt.figure()  # new figure
        # plt.suptitle('Loss by Test Period, Tree Max Depth, N Trees')  # overlays the subplots
        loss_metric = 'rmse'
        loss_metric = 'mae'
        axes_number = 0
        n_months_backs = range(1, 7, 1)
        last_test_period_index = len(test_periods) - 1
        last_n_months_back_index = len(n_months_backs) - 1
        for test_period_index, test_period in enumerate(test_periods):
            for n_months_back_index, n_months_back in enumerate(n_months_backs):
                axes_number += 1  # count across rows
                plt.subplot(len(test_periods), len(n_months_backs), axes_number)
                make_subplot(test_period, n_months_back, loss_metric)
                if test_period_index == last_test_period_index:
                    # annotate the bottom row only
                    if n_months_back_index == 0:
                        plt.xlabel('alpha-l1_ratio')
                        plt.ylabel('%s x $1000' % loss_metric)
                    if n_months_back_index == last_n_months_back_index:
                        plt.legend(loc='best', fontsize=5)

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        out_suffix = '-%02d' % months if len(months) == 1 else ''
        plt.savefig(control.path_chart_base + str(year) + out_suffix + '.pdf')
        plt.close()

    for year in (2004, 2005, 2006, 2007, 2008, 2009):
        months = (2,) if year == 2009 else (2, 5, 8, 11)
        for month in months:
            make_figure(year, (month,))
        make_figure(year, months)
        if control.test:
            break


def make_charts(df, control, ege_control):
    'write one txt file for each n_months_back'
    def make_subplot(year, month, n_months_back):
        'mutate the default axes'
        test_period = str(year * 100 + month)  # yyyymm
        mask = (
            (df.test_period == test_period) &
            (df.n_months_back == n_months_back)
        )
        subset = df.loc[mask]
        y = subset.mae
        y.index = subset.learning_rate.values
        y = y.sort_index()
        plt.plot(y / 1000.0)
        x_label = ['%4.2f' % x for x in y.index]
        plt.xticks(range(len(y)), x_label, size='xx-small', rotation='vertical')
        plt.yticks(size='xx-small')
        plt.title('%4d-%02d %1d' % (year, month, n_months_back),
                  loc='right',
                  fontdict={'fontsize': 'xx-small', 'style': 'italic'},
                  )
        return
        learning_rates = sorted(set(df.learning_rates))
        n_ticks = len(learning_rates)
        # OLD BELOW ME
        l1_ratios = sorted(set(df.l1_ratio))
        n_ticks = len(l1_ratios)
        line_index = -1
        for units_X in ('natural', 'log'):
            for units_y in ('natural', 'log'):
                line_index += 1
                x_label = []
                y = np.empty((n_ticks,), dtype=float)
                tick_index = -1
                for l1_ratio in l1_ratios:
                    tick_index += 1
                    mask = (
                        (df.test_period == test_period) &
                        (df.n_months_back == n_months_back) &
                        (df.units_X == units_X) &
                        (df.units_y == units_y) &
                        (df.alpha == alpha) &
                        (df.l1_ratio == l1_ratio)
                    )
                    if sum(mask) != 1:
                        # something is wrong
                        print 'mask', sum(mask)
                        print 'test_period', sum(df.test_period == test_period)
                        print 'n_months_back', sum(df.n_months_back == n_months_back)
                        print 'units_X', sum(df.units_X == units_X)
                        print 'units_y', sum(df.units_y == units_y)
                        print 'alpha', sum(df.alpha == alpha)
                        print 'l1_ratio', sum(df.l1_ratio == l1_ratio)
                        print test_period, n_months_back, units_X, units_y, alpha, l1_ratio
                        pdb.set_trace()
                    subset = df.loc[mask]
                    assert len(subset) == 1, subset.shape
                    row = dict(subset.iloc[0])
                    y[tick_index] = row[loss_metric]
                    x_label.append('%4.2f' % row['l1_ratio'])
                plt.plot(y / 1000.0,
                         label='%3s-%3s' % (units_X, units_y),
                         linestyle=[':', '-.', '--', '-'][line_index % 4],
                         color='bgrcmykw'[line_index % 8],
                         )
                plt.xticks(range(len(y)), x_label, size='xx-small', rotation='vertical')
                plt.yticks(size='xx-small')
        plt.title('%4.2f %s-%s %d' % (alpha, test_period[:4], test_period[4:], n_months_back),
                  loc='left',
                  fontdict={'fontsize': 'xx-small', 'style': 'italic'},
                  )

    def make_figure(year):

        print 'make_figure', year
        plt.figure()  # new figure
        # plt.suptitle('Loss by Test Period, Tree Max Depth, N Trees')  # overlays the subplots
        axes_number = 0
        n_months_backs = (1, 2, 3, 4, 5, 6)
        month_seq = (2, 5, 8, 11)
        for month in month_seq:
            for n_months_back in n_months_backs:
                axes_number += 1  # count across rows
                plt.subplot(len(month_seq), len(n_months_backs), axes_number)
                make_subplot(year, month, n_months_back)
                # annotate the bottom row only
                if month == 11:
                    if n_months_back == 1:
                        plt.xlabel('learning_rate')
                        plt.ylabel('mae x $1000')
                    if n_months_back == 6:
                        plt.legend(loc='best', fontsize=5)

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(control.path_chart_base + str(year) + '.pdf')
        plt.close()

    for year in (2004, 2005, 2006, 2007, 2008, 2009):
        make_figure(year)
        if control.test:
            break
    return


def errors(actuals, predictions):
    'return root_mean_squared_error, median_absolute_error'
    errors = actuals - predictions
    root_mean_squared_error = np.sqrt(np.sum(errors * errors) / (1.0 * len(errors)))
    median_absolute_error = np.median(np.abs(errors))
    return root_mean_squared_error, median_absolute_error


def extract_yyyymm(path):
    return path.split('/')[4].split('.')[0]


def make_data(control):
    'return data frame, ege_control'
    def process_file(path, rows_list):
        'mutate rows_list, a list of dictionaries, to include objects at path'
        print 'reducing', path
        with open(path, 'rb') as f:
            val_result, ege_control = pickle.load(f)
        yyyymm = extract_yyyymm(path)
        for k, v in val_result.iteritems():
            is_en = isinstance(k, ResultKeyEn)
            is_gbr = isinstance(k, ResultKeyGbr)
            is_rfr = isinstance(k, ResultKeyRfr)
            is_tree = is_gbr or is_rfr
            actuals = v.actuals.values
            predictions = v.predictions
            if predictions is None:
                print k
                print 'predictions is missing'
                pdb.set_trace()
            rmse, mae = errors(actuals, predictions)
            row = {
                'yyyymm': yyyymm,
                'n_months_back': k.n_months_back,
                'units_X': k.units_X,
                'units-y': k.units_y,
                'alpha': k.alpha if is_en else None,
                'l1_ratio': k.l1_ratio if is_en else None,
                'n_estimators': k.n_estimators if is_tree else None,
                'max_features': k.max_features if is_tree else None,
                'max_depth': k.max_depth if is_tree else None,
                'loss': k.loss if is_gbr else None,
                'learning_rate': k.learning_rate if is_gbr else None,
                'mae': mae,
            }
            rows_list.append(row)
        return ege_control  # return last ege_control value (they should all be the same)

    rows_list = []
    for path in glob.glob(control.path_in_ege):
        ege_control = process_file(path, rows_list)
        if control.test:
            break
    df = pd.DataFrame(rows_list)
    return df, ege_control  # return last ege_control, not all


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(base_name=control.arg.base_name)
    print control

    if control.arg.data:
        df, ege_control = make_data(control)
        with open(control.path_reduction, 'wb') as f:
            pickle.dump((df, ege_control, control), f)
    else:
        with open(control.path_reduction, 'rb') as f:
            df, ege_control, data_control = pickle.load(f)
        make_charts(df, control, ege_control)

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
        ResultKey
        ResultValue

    main(sys.argv)
