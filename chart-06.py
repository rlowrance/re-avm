'''create charts showing results of valgbr.py

INVOCATION
  python chart-06.py [--data] [--test]

INPUT FILES
 INPUT/valavm/YYYYMM.pickle

OUTPUT FILES
 jWORKING/chart-06/[test-]data.pickle   | reduced data
 WORKING/chart-06/[test-]a.pdf          | range of losses by model (graph)
 WORKING/chart-06/[test-]b-MM.pdf       | HPs with lowest losses
 WORKING/chart-06/[test-]c.pdf          | best model each month
 WORKING/chart-06/[test-]d.pdf          | best & 50th best each month
 WORKING/chart-06/[test-]e.pdf          | best 50 models each month (was chart-07)
 WORKING/chart-06/best.pickle         dataframe with best choices each month
where
 TESTMONTH is YYYYMM
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
# from chart_01_datakey import DataKey
from columns_contain import columns_contain
from Logger import Logger
from ParseCommandLine import ParseCommandLine
from Path import Path
from Report import Report
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
        path_in_ege=dir_working + 'valavm/??????.pickle',
        path_in_samples='../data/working/samples-train.csv',
        path_reduction=dir_path + reduced_file_name,
        path_median_prices=dir_working + 'chart-01/data.pickle',
        path_chart_base=dir_path,
        random_seed=random_seed,
        test=arg.test,
    )


def select_and_sort(df, year, month, model):
    'return new df contain sorted observations for specified year, month, model'
    yyyymm = str(year * 100 + month)
    mask = (
        (df.model == model) &
        (df.yyyymm == yyyymm)
    )
    subset = df.loc[mask]
    assert len(subset) > 0, subset.shape
    return subset.sort_values('mae')


def make_chart_a(reduction, control):
    'graph range of errors by month by method'
    print 'make_chart_a'

    def make_subplot(year, month):
        'mutate the default axes'
        print 'make_subplot month', month
        for model in set(reduction.model):
            subset_sorted = select_and_sort(reduction, year, month, model)
            y = (subset_sorted.mae / 1000.0)
            plt.plot(
                y,
                label=model,
            )
            plt.yticks(size='xx-small')
            plt.title(
                '%4d-%02d' % (year, month),
                loc='right',
                fontdict={'fontsize': 'xx-small', 'style': 'italic'},
            )
            plt.xticks([])  # no ticks on x axis
        return

    def make_figure():
        # make and save figure

        plt.figure()  # new figure
        # plt.suptitle('Loss by Test Period, Tree Max Depth, N Trees')  # overlays the subplots
        axes_number = 0
        year_months = ((2006, 12), (2007, 1), (2007, 2),
                       (2007, 3), (2007, 4), (2007, 5),
                       (2007, 6), (2006, 7), (2006, 8),
                       (2007, 9), (2007, 10), (2007, 11),
                       )
        row_seq = (1, 2, 3, 4)
        col_seq = (1, 2, 3)
        for row in row_seq:
            for col in col_seq:
                year, month = year_months[axes_number]
                axes_number += 1  # count across rows
                plt.subplot(len(row_seq), len(col_seq), axes_number)
                make_subplot(year, month)
                # annotate the bottom row only
                if row == 4:
                    if col == 1:
                        plt.xlabel('hp set')
                        plt.ylabel('mae x $1000')
                    if col == 3:
                        plt.legend(loc='best', fontsize=5)

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(control.path_chart_base + str(year) + '-a.pdf')
        plt.close()

    make_figure()
    return


def differ_only_in_units(a, b):
    if a is None:
        return False
    # print a, b
    for index, value in a.iteritems():
        # print index, value
        if index == 'units_y':
            continue
        if index == 'units_X':
            continue
        if isinstance(value, float) and np.isnan(value) and np.isnan(b[index]):
            continue
        if value != b[index]:
            return True
    return False


def differ(a, b):
    if a is None:
        return b is None
    for index, value in a.iteritems():
        if isinstance(value, float) and np.isnan(value) and (not np.isnan(b[index])):
            return True
        if value != b[index]:
            return True
    return False


def make_chart_b(reduction, control):
    '''MAE for best-performing models by training period
    NOTE: This code knows that en is never the best model
    '''
    def report(year, month):
        format_header = '%6s %5s %2s %3s %4s %4s %-20s'
        format_detail = '%6d %5s %2d %3d %4d %4s %-20s'
        n_detail_lines = 50

        def header(r):
            r.append('MAE for %d best-performing models and their hyperparameters' % n_detail_lines)
            r.append('Training period: %d-%0d' % (year, month))
            r.append(' ')
            r.append(format_header % ('MAE', 'Model', 'bk', 'mxd', 'nest', 'mxft', 'Other HPs'))

        def append_detail_line(r, series):
            'append detail line to report r'
            hp_string = ''
            # accumulate non-missing hyperparameters
            called_out_indices = set(('mae', 'model', 'yyyymm', 'n_months_back',
                                      'max_depth', 'n_estimators', 'max_features'))
            for index, value in series.iteritems():
                print index, value
                if index in called_out_indices:
                    # not a hyperparameter collected into "Other HPs"
                    continue
                if (isinstance(value, float) and np.isnan(value)) or value is None:
                    # value is missing
                    continue
                if index == 'units_X' and value == 'natural':
                    continue
                if index == 'units_y' and value == 'natural':
                    continue
                hp_string += '%s=%s ' % (index, value)
            assert series.model != 'en', series
            r.append(format_detail % (
                series.mae, series.model, series.n_months_back,
                series.max_depth,
                series.n_estimators,
                series.max_features,
                hp_string))

        def footer(r):
            r.append(' ')
            r.append('column legend:')
            r.append('bk -> number of months back for training')
            r.append('mxd -> max depth of individual tree')
            r.append('nest -> n_estimators (number of trees)')
            r.append('mxft -> max number of features considered when selecting new split variable')

        yyyymm = str(year * 100 + month)
        mask = (
            reduction.yyyymm == yyyymm
        )
        subset = reduction.loc[mask]
        assert len(subset) > 0, subset.shape
        subset_sorted = subset.sort_values('mae')
        r = Report()
        header(r)
        detail_line_number = 0
        previous_row_series = None
        for row in subset_sorted.iterrows():
            print row
            row_series = row[1]
            print previous_row_series, row_series
            detail_line_number += 1
            append_detail_line(r, row_series)
            previous_row_series = row_series
            if detail_line_number == n_detail_lines:
                break
        footer(r)
        r.write(control.path_chart_base + yyyymm + '-b.txt')

    for year in (2006, 2007):
        months = (12,) if year == 2006 else (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
        for month in months:
            report(year, month)


def test_months():
    return (200612, 200701, 200702, 200703, 200704, 200705,
            200706, 200707, 200708, 200709, 200710, 200711,
            )


def make_best_dictOLD(reduction, median_prices):
    'return dict st key=test_date, value=(median_price, sorted_hps)'
    best = {}
    median_prices = {}
    for test_month in test_months():
        mask = reduction.test_month == test_month.as_str()
        subset = reduction.loc[mask]
        subset_sorted = subset.sort_values('mae')
        best[test_month] = (median_prices[test_month.as_int()], subset_sorted)
    return best


class ChartReport(object):
    def __init__(self):
        self.report = Report()
        self.format_header = '%6s %4s %6s %8s %5s %2s %3s %4s %4s %-20s'
        self.format_detail = '%6d %4d %6d %8d %5s %2d %3d %4d %4s %-20s'
        self._header()

    def append(self, line):
        self.report.append(line)

    def write(self, path):
        self._footer()
        self.report.write(path)

    def _header(self):
        self.report.append('Median Absolute Error (MAE) by month for best-performing models and their hyperparameters')
        self.report.append(' ')
        self.report.append(self.format_header % ('Test', '', '', 'Median', '', '', '', '', '', ''))
        self.report.append(self.format_header % (
            'Month', 'rank', 'MAE', 'Price',
            'Model', 'bk', 'mxd', 'nest', 'mxft', 'Other HPs'))

    def _footer(self):
        self.report.append(' ')
        self.report.append('column legend:')

        def legend(tag, meaning):
            self.report.append('%12s -> %s' % (tag, meaning))

        legend('Test Month', 'year-month')
        legend('rank', 'rank within month; 1 ==> best/lowest MAE')
        legend('MAE', 'median absolute error in the price estimate')
        legend('Median Price', 'median price in the test month')
        legend('bk', 'number of months back for training')
        legend('mxd', 'max depth of individual tree')
        legend('nest', 'n_estimators (number of trees)')
        legend('mxft', 'max number of features considered when selecting new split variable')

    def detail_line(self, sorted_index, test_month, sorted_df, median_price):
        def make_hp_string(series):
            'accumulate HPs not printed in columns'
            # the common HPs are printed in columns
            hp_string = ''
            called_out_indices = set(('mae', 'model', 'yyyymm', 'n_months_back',
                                      'max_depth', 'n_estimators', 'max_features',
                                      'test_month',
                                      ))
            for index, value in series.iteritems():
                if index in called_out_indices:
                    # not a hyperparameter collected into "Other HPs"
                    continue
                if (isinstance(value, float) and np.isnan(value)) or value is None:
                    # value is missing
                    continue
                if series.model != 'en' and index == 'units_X' and value == 'natural':
                    continue
                if series.model != 'en' and index == 'units_y' and value == 'natural':
                    continue
                hp_string += '%s=%s ' % (index, value)
            return hp_string

        assert len(sorted_df) > 0, (test_month, sorted_index)
        series = sorted_df.iloc[sorted_index]
        if series.model == 'en':
            self.report.append(self.format_detail % (
                test_month, sorted_index + 1,
                series.mae, median_price,
                series.model, series.n_months_back,
                0, 0, 0,   # these are tree-oriented HPs
                make_hp_string(series)))
        else:
            self.report.append(self.format_detail % (
                test_month, sorted_index + 1,
                series.mae, median_price,
                series.model, series.n_months_back,
                series.max_depth,
                series.n_estimators,
                series.max_features,
                make_hp_string(series)))


def make_chart_cd(reduction, control, median_prices, sorted_hps, detail_lines, report_id):
    '''write report: mae, model, HPs for month'''
    cr = ChartReport()
    for test_month in test_months():
        median_price = median_prices[test_month]
        sorted_hps_test_month = sorted_hps[test_month]
        for dl in detail_lines(sorted_hps_test_month):
            cr.detail_line(dl, test_month, sorted_hps_test_month, median_price)
    cr.write(control.path_chart_base + report_id + '.txt')
    return


def make_median_prices(medians):
    return {test_month: medians[test_month.as_int()]
            for test_month in test_months()
            }


def make_charts(reduction, samples, control, median_prices):
    # make_chart_a(reduction, control)
    # make_chart_b(reduction, control)
    def make_sorted_hps(reduction):
        'return dict; key = test_month; value = df of HPs, sorted by MAE'
        def subset_sorted(test_month):
            mask = reduction.test_month == str(test_month)
            return reduction.loc[mask].sort_values('mae')

        return {test_month: subset_sorted(test_month)
                for test_month in test_months()
                }

    sorted_hps = make_sorted_hps(reduction)

    def detail_lines_c(sorted_hps_test_month):
        return [0]  # the one with the lowest MAE (the best set of HPs)

    def detail_lines_d(sorted_hps_test_month):
        result = []
        for k in xrange(len(sorted_hps_test_month)):
            result.append(k)
            if k == 4:
                break
        result.append(len(sorted_hps_test_month) - 1)
        return result

    make_chart_cd(reduction, control, median_prices, sorted_hps, detail_lines_c, 'c')
    make_chart_cd(reduction, control, median_prices, sorted_hps, detail_lines_d, 'd')


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
    def make_row(test_month, k, v):
        is_en = isinstance(k, ResultKeyEn)
        is_gbr = isinstance(k, ResultKeyGbr)
        is_rfr = isinstance(k, ResultKeyRfr)
        is_tree = is_gbr or is_rfr
        model = 'en' if is_en else ('gb' if is_gbr else 'rf')
        actuals = v.actuals.values
        predictions = v.predictions

        if predictions is None:
            print k
            print 'predictions is missing'
            pdb.set_trace()
        rmse, mae = errors(actuals, predictions)
        return {
            'model': model,
            'test_month': test_month,
            'n_months_back': k.n_months_back,
            'units_X': k.units_X if is_en else 'natural',
            'units_y': k.units_y if is_en else 'natural',
            'alpha': k.alpha if is_en else None,
            'l1_ratio': k.l1_ratio if is_en else None,
            'n_estimators': k.n_estimators if is_tree else None,
            'max_features': k.max_features if is_tree else None,
            'max_depth': k.max_depth if is_tree else None,
            'loss': k.loss if is_gbr else None,
            'learning_rate': k.learning_rate if is_gbr else None,
            'mae': mae,
        }

    def append_rows(path, rows_list):
        'mutate rows_list, a list of dictionaries, to include objects at path'
        print 'reducing', path
        test_month = extract_yyyymm(path)
        n = 0
        with open(path, 'rb') as f:
            while True:
                try:
                    record = pickle.load(f)  # read until EOF
                    assert isinstance(record, tuple), type(record)
                    key, value = record
                    n += 1
                    rows_list.append(make_row(test_month, key, value))
                except ValueError as e:
                    if key is not None:
                        print key
                    if key is not None:
                        print value
                    print 'ValueError', e  # ignore error
                except EOFError:
                    break
        print 'number of records', n

    rows_list = []
    paths = sorted(glob.glob(control.path_in_ege))
    for path in paths:
        append_rows(path, rows_list)
        if control.test:
            break
    return pd.DataFrame(rows_list)


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(base_name=control.arg.base_name)
    print control

    if control.arg.data:
        df = make_data(control)
        with open(control.path_reduction, 'wb') as f:
            pickle.dump((df, control), f)
    else:
        samples = pd.read_csv(control.path_in_samples)
        with open(control.path_reduction, 'rb') as f:
            reduction, reduction_control = pickle.load(f)
            with open(control.path_median_prices, 'rb') as g:
                data, reduction_control = pickle.load(g)
                counts, means, medians, prices = data
                make_charts(reduction, samples, control, medians)

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
        ResultValue

    main(sys.argv)
