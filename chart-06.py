'''create charts showing results of valgbr.py

INVOCATION
  python chart-06.py [--data] [--test VALAVM]
where
  VALAVM in {roy, anil}

INPUT FILES
 WORKING/chart-01/data.pickle
 WORKING/samples-train.csv
 WORKING/valavm[-VALAVM-test]/YYYYMM.pickle

OUTPUT FILES
 WORKING/chart-06[-VALAVM]/data.pickle    | reduced data
 WORKING/chart-06[-VALAVM]/a.pdf          | range of losses by model (graph)
 WORKING/chart-06[-VALAVM]/b-YYYYMM.txt   | HPs with lowest losses
 WORKING/chart-06[-VALAVM]/c.pdf          | best model each month
 WORKING/chart-06[-VALAVM]/d.pdf          | best & 50th best each month
 WORKING/chart-06[-VALAVM]/e.pdf          | best 50 models each month (was chart-07)
 WORKING/chart-06[-VALAVM]/best.pickle    | dataframe with best choices each month
 WORKING/chart-06[-VALAVM]/log[-data].txt | log file (created by print statements)
'''

from __future__ import division

import argparse
import collections
import cPickle as pickle
import glob
import math
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
from Month import Month
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

ModelResult = collections.namedtuple('ModelResult', 'mae')


def make_control(argv):
    # return a Bunch

    print argv
    parser = argparse.ArgumentParser()
    parser.add_argument('invocation')
    parser.add_argument('--data', action='store_true')
    parser.add_argument('--test')  # arg.test is None or a str
    arg = parser.parse_args(argv)  # arg.__dict__ contains the bindings
    arg.base_name = 'chart-06'

    random_seed = 123
    random.seed(random_seed)

    debug = False

    # assure output directory exists
    dir_working = Path().dir_working()
    dir_out = (
        dir_working +
        'chart-06' +
        ('' if arg.test is None else ('-' + arg.test)) +
        '/'
    )
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    return Bunch(
        arg=arg,
        debug=debug,
        path_in_ege=(
            dir_working +
            'valavm' +
            ('' if arg.test is None else ('-' + arg.test + '-test')) +
            '/??????.pickle'),
        path_in_samples=dir_working + 'samples-train.csv',
        path_out_a=dir_out + 'a.pdf',
        path_out_b=dir_out + 'b-%d.txt',
        path_out_cd=dir_out + '%s.txt',
        path_out_d=dir_out + 'd.txt',
        path_out_e=dir_out + 'e-%04d-%6d.txt',
        path_out_f=dir_out + 'f-%04d.txt',
        path_out_g=dir_out + 'g.txt',
        path_data=dir_out + 'data.pickle',
        path_out_best=dir_out + 'best.pickle',
        path_out_log=dir_out + 'log' + ('-data' if arg.data else '') + '.txt',
        path_in_chart_01_reduction=dir_working + 'chart-01/data.pickle',
        random_seed=random_seed,
        test=arg.test,
    )


def select_and_sort(df, year, month, model):
    'return new df contain sorted observations for specified year, month, model'
    yyyymm = str(year * 100 + month)
    mask = (
        (df.model == model) &
        (df.test_month == yyyymm)
    )
    subset = df.loc[mask]
    if len(subset) == 0:
        print 'empty subset'
        print year, month, model, sum(df.model == model), sum(df.test_month == yyyymm)
        pdb.set_trace()
    return subset.sort_values('mae')


def make_chart_a_OLD(reduction, control):
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
                       (2007, 6), (2007, 7), (2007, 8),
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
        plt.savefig(control.path_out_a)
        plt.close()

    make_figure()
    return


def make_chart_a(reduction, control):
    'graph range of errors by month by method'
    print 'make_chart_a'
    pdb.set_trace()

    def make_subplot(year, month):
        'mutate the default axes'
        print 'make_subplot month', month
        pdb.set_trace()
        for model in ('en', 'rf', 'gb'):
            y = reduction.select_maes_model_year_month(model, year, month).sort()
            plt.plot(y, label=model)
            plt.yticks(size='xx-small')
            plt.title(
                '%4d-%02d' % (year, month),
                loc='right',
                fontdict={'fontsize': 'xx-small', 'style': 'italic'}
            )
            plt.xticks([])  # no ticks on x axis
        return
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
                       (2007, 6), (2007, 7), (2007, 8),
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
        plt.savefig(control.path_out_a)
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


def make_hp_string(series):
    ignored_indices = set(('mae', 'model', 'yyyymm', 'n_months_back',
                           'max_depth', 'n_estimators', 'max_features',
                           'test_month', 'loss'))
    result = ''
    for index, value in series.iteritems():
        # check that index == 'loss' has no information
        assert (value in ('ls', None) if index == 'loss' else True), (index, value)
        if index in ignored_indices:
            continue
        if (isinstance(value, float) and np.isnan(value)) or value is None:
            # value is missing
            continue
        if index == 'units_X' and value == 'natural':
            continue
        if index == 'units_y' and value == 'natural':
            continue
        result += '%s=%s ' % (index, value)
    return result


def make_chart_b(reduction, control):
    '''MAE for best-performing models by training period
    NOTE: This code knows that en is never the best model '''
    def report(year, month):
        print 'make_chart_b::report', year, month
        format_header = '%6s %5s %2s %4s %4s %4s %-20s'
        format_detail = '%6d %5s %2d %4d %4d %4s %-20s'
        format_detail_no_max_depth = '%6d %5s %2d %4s %4d %4s %-20s'

        n_detail_lines = 50

        def header(r):
            r.append('MAE for %d best-performing models and their hyperparameters' % n_detail_lines)
            r.append('Validation month: %d-%0d' % (year, month))
            r.append(' ')
            r.append(format_header % ('MAE', 'Model', 'bk', 'mxd', 'nest', 'mxft', 'Other HPs'))

        def append_detail_line(r, series):
            'append detail line to report r'
            hp_string = make_hp_string(series)
            # accumulate non-missing hyperparameters
            assert series.model != 'en', series
            if series.max_depth is None:
                r.append(format_detail_no_max_depth % (
                    series.mae, series.model, series.n_months_back,
                    'None',
                    series.n_estimators,
                    series.max_features,
                    hp_string))
            else:
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
            reduction.test_month == yyyymm
        )
        subset = reduction.loc[mask]
        assert len(subset) > 0, (subset.shape, yyyymm)
        subset_sorted = subset.sort_values('mae')
        r = Report()
        header(r)
        detail_line_number = 0
        for row in subset_sorted.iterrows():
            # print row
            row_series = row[1]
            detail_line_number += 1
            append_detail_line(r, row_series)
            if detail_line_number == n_detail_lines:
                break
        footer(r)
        r.write(control.path_out_b % int(yyyymm))

    for year in (2006, 2007):
        months = (12,) if year == 2006 else (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
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


class ChartCDReport(object):
    def __init__(self):
        self.report = Report()
        self.format_header = '%6s %4s %6s %8s %5s %2s %4s %4s %4s %-20s'
        self.format_detail = '%6d %4d %6d %8d %5s %2d %4d %4d %4s %-20s'
        self.format_detail_no_max_depth = '%6d %4d %6d %8d %5s %2d %4s %4d %4s %-20s'
        self._header()

    def append(self, line):
        self.report.append(line)

    def write(self, path):
        self._footer()
        self.report.write(path)

    def _header(self):
        self.report.append('Median Absolute Error (MAE) by month for best-performing models and their hyperparameters')
        self.report.append(' ')
        self.report.append(self.format_header % ('Vald.', '', '', 'Median', '', '', '', '', '', ''))
        self.report.append(self.format_header % (
            'Month', 'rank', 'MAE', 'Price',
            'Model', 'bk', 'mxd', 'nest', 'mxft', 'Other HPs'))

    def _footer(self):
        self.report.append(' ')
        self.report.append('column legend:')

        def legend(tag, meaning):
            self.report.append('%12s -> %s' % (tag, meaning))

        legend('Vald. Month', 'validation month in format year-month')
        legend('rank', 'rank within month; 1 ==> best/lowest MAE')
        legend('MAE', 'median absolute error in the price estimate')
        legend('Median Price', 'median price in the test month')
        legend('bk', 'number of months back for training')
        legend('mxd', 'max depth of individual tree')
        legend('nest', 'n_estimators (number of trees)')
        legend('mxft', 'max number of features considered when selecting new split variable')

    def detail_line(self, sorted_index, test_month, sorted_df, median_price):
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
            if series.max_depth is None:
                self.report.append(self.format_detail_no_max_depth % (
                    test_month, sorted_index + 1,
                    series.mae, median_price,
                    series.model, series.n_months_back,
                    'None',
                    series.n_estimators,
                    series.max_features,
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
    r = ChartCDReport()
    for test_month in test_months():
        median_price = median_prices[test_month]
        sorted_hps_test_month = sorted_hps[test_month]
        for dl in detail_lines(sorted_hps_test_month):
            r.detail_line(dl, test_month, sorted_hps_test_month, median_price)
    r.write(control.path_out_cd % report_id)
    return


def make_median_prices(medians):
    return {test_month: medians[test_month.as_int()]
            for test_month in test_months()
            }


class ChartEReport(object):
    def __init__(self, k, ensemble_weighting, valavm):
        self.report = Report()
        self.format_header = '%6s %20s %5s %6s %6s %6s'
        self.format_model = '%6d %20s %5d %6.0f %6.4f %6.0f'
        self.format_best = '%6d %20s %5d %6.0f %6s %6.0f best'
        self.format_ensemble = '%6s %20s %5s %6s %6s %6.0f'
        self.format_relative_error = '%6s %20s %5s %6s %6s %5.3f%%'
        self.format_marr = '%6s %20s %5s %6s %6s %6.2%%'
        self._header(k, ensemble_weighting, valavm)

    def _header(self, k, ensemble_weighting, valavm):
        self.report.append('Performance of Best Models Separately and as an Ensemble')
        self.report.append(' ')
        self.report.append('Consider best K = %d models' % k)
        self.report.append('Ensemble weighting: %s' % ensemble_weighting)
        self.report.append('Hyperparameter set: %s' % 'all' if valavm == 'roy' else valavm)

        self.report.append(' ')
        self.report.append(self.format_header % ('vald.', ' ', 'rank', 'vald.', '', 'next'))
        self.report.append(self.format_header % ('month', 'model', 'index', 'MAE', 'weight', 'MAE'))

    def model_detail(self, month, model_s, rank, test_mae, weight, next_mae):
        self.report.append(self.format_model % (month, model_s, rank, test_mae, weight, next_mae))

    def model_best(self, month, model_s, rank, test_mae, next_mae):
        self.report.append(self.format_best % (month, model_s, rank, test_mae, ' ', next_mae))

    def ensemble_detail(self, ensemble_mae):
        self.report.append(self.format_ensemble % (' ', 'ensemble', 'na', 'na', 'na', ensemble_mae))

    def marr(self, amount):
        self.report.append(self.format_relative_error % (' ', 'med abs rel regret', 'na', 'na', 'na', amount * 100.0))

    def median_price(self, amount):
        self.report.append(self.format_ensemble % (' ', 'median price', 'na', 'na', 'na', amount))

    def regret(self, amount):
        self.report.append(self.format_ensemble % (' ', 'regret', 'na', 'na', 'na', amount))

    def relative_error(self, amount):
        self.report.append(self.format_relative_error % (' ', 'relative regret', 'na', 'na', 'na', amount * 100.0))

    def write(self, path):
        'append legend and write the file'
        def a(line):
            self.report.append(line)

        a(' ')
        a('Legend for columns:')
        a('vald. month --> validation month for selecting models')
        a('               parameters are trained starting in bk prior month')
        a('               where bk is part of the best model description')
        a('model      --> description of the model (see below)')
        a('rank index --> performance of model on test month data')
        a('               the best model has index 0')
        a('               the second best model has index 1')
        a('vald. MAE   --> the median absolute error of the model in the validation month')
        a('weight     --> the weight of the model in the ensemble method')
        a('next MAE   --> the median absolute error of the model in the month after the test month')
        a(' ')
        a('Legend for model descriptions:')
        a('model descriptions is one of')
        a('rf-A-B-C          --> model is random forests, where')
        a('                    A = number of months of training data used')
        a('                        example: A = 3 for test month 200612 means')
        a('                        that the model was trained on data from months')
        a('                        200609, 200610, and 200612')
        a('                    B = the n_estimators value from scikit learn, which is')
        a('                        the number of trees in the forest')
        a('                    C = the max_features value from scikit learn, which is')
        a('                        the number of features considered when splitting a')
        a('                        node in the tree, where')
        a('                        auto means use all the features when splitting a node')
        a('                        sqrt means use the sqrt of the number of features,')
        a('                        log2 means use the log_s of the number of features')
        a('gb-A-B-D-E      --> model is gradient boosting,  where')
        a('                    D = max_depth, maximum depth of an individual tree')
        a('                    E = the learning rate for shrinking the contribution of')
        a('                        the next iteration')
        a('ensemble        --> next MAE contains median absolute error of the ensemble')
        a('regret          --> (MAE of best model in next month) - (MAE of the ensemble)')
        a('median price    --> the median price in the next month after the test period')
        a('relative regret --> next MAE contains regret / median price in the next month')

        self.report.write(path)

    def append(self, line):
        self.report.append(line)


class ChartFReport(object):
    def __init__(self, k, ensemble_weighting, valavm):
        self.report = Report()
        self.format_header = '%6s %20s %6s'
        self.format_ensemble = '%6d %20s  %6.0f'
        self.format_relative_error = '%6d %20s %5.3f%%'
        self.format_regret = '%6d %20s %6.0f'
        self.format_marr = '%6s %20s %5.3f%%'
        self._header(k, ensemble_weighting, valavm)

    def _header(self, k, ensemble_weighting, valavm):
        self.report.append('Regret of Ensemble Models')
        self.report.append(' ')
        self.report.append('Consider best %d models' % k)
        self.report.append('Ensemble weighting: %s' % ensemble_weighting)
        self.report.append('Hyperparameter set: %s' % 'all' if valavm == 'roy' else valavm)
        self.report.append(' ')
        self.report.append(
            self.format_header % (
                'vald.', ' ', 'next'))
        self.report.append(
            self.format_header % (
                'month', 'model', 'MAE'))

    def ensemble_detail(self, test_month, mae):
        self.report.append(
            self.format_ensemble % (
                test_month, 'ensemble', mae))

    def median_price(self, test_month, amount):
        self.report.append(
            self.format_ensemble % (
                test_month, 'median price', amount))

    def regret(self, test_month, amount):
        self.report.append(
            self.format_regret % (
                test_month, 'regret', amount))

    def relative_error(self, test_month, amount):
        self.report.append(
            self.format_relative_error % (
                test_month, 'relative regret', amount * 100.0))

    def marr(self, amount):
        self.report.append(
            self.format_marr % (
                ' ', 'med abs rel regret', amount * 100.0))

    def append(self, line):
        self.report.append(line)

    def write(self, path):
        'append legend and write the file'
        def a(line):
            self.report.append(line)

        a(' ')
        a('Legend for columns:')
        a('vald. month --> validation month for selecting models')
        a('                parameters are trained starting in bk prior month')
        a('                where bk is part of the best model description')
        a('model       --> description of the model (see below)')
        a('rank index  --> performance of model on test month data')
        a('                the best model has index 0')
        a('                the second best model has index 1')
        a('test MAE    --> the median absolute error of the model in the test month')
        a('weight      --> the weight of the model in the ensemble method')
        a('next MAE    --> the median absolute error of the model in the month after the test month')

        self.report.write(path)


def model_to_str(k):
    if isinstance(k, ResultKeyEn):
        return 'en-%d-%s-%s-%d-%d' % (
            k.n_months_back, k.units_X, k.units_y, k.alpha, k.l1_ratio)
    elif isinstance(k, ResultKeyGbr):
        print k
        assert k.loss == 'ls', k
        return 'gb-%d-%d-%s-%.1f' % (
            k.n_months_back, k.n_estimators, str(k.max_depth), k.learning_rate)
    elif isinstance(k, ResultKeyRfr):
        if (k.n_estimators is None) or (k.n_months_back is None):
            print k
            pdb.set_trace()
        return 'rf-%d-%d-%s-%s' % (
            k.n_months_back, k.n_estimators, str(k.max_depth), str(k.max_features))
    else:
        print type(k)
        pdb.set_trace()
    return 'model'


def check_actuals(actuals):
    'each should be the same'
    k = len(actuals)
    assert k > 0, k
    first = actuals[0]
    for other in actuals:
        if collections.Counter(first) != collections.Counter(other):
            print collections.Counter(first), collections.Counter(other)
            pdb.set_trace()


def make_ensemble_predictions(predictions, weights):
    'return vector of predictions: sum w_i pred_i / sum w_i'
    sum_weighted_predictions = np.array(predictions[0])
    sum_weighted_predictions.fill(0.0)
    for index in xrange(len(weights)):
        sum_weighted_predictions = np.add(
            sum_weighted_predictions,
            np.dot(predictions[index], weights[index]))
    sum_weights = np.sum(np.array(weights))
    result = sum_weighted_predictions / sum_weights
    return result


def make_charts_ef(k, actuals_d, predictions_d, mae_d, control, median_prices):
    '''Write charts e and f, return median-absolute-relative_regret object'''
    if len(actuals_d) != len(predictions_d) != len(mae_d):
        pdb.trace()
    ensemble_weighting = 'exp(-MAE/100000)'
    relative_errors = []
    f = ChartFReport(k, ensemble_weighting, control.arg.valavm)
    for test_month in test_months():
        e = ChartEReport(k, ensemble_weighting, control.arg.valavm)
        test_month_mae = mae_d[test_month]
        models_sorted_test_month = sorted(test_month_mae, key=test_month_mae.get)  # keys in order of mae
        weights = []
        actuals = []
        predictions = []
        next_month = Month(test_month).increment(1).as_int()
        for index in xrange(k):
            model = models_sorted_test_month[index]
            test_mae = mae_d[test_month][model]
            next_mae = mae_d[next_month][model]
            weight = math.exp(-test_mae / 100000.0)
            e.model_detail(test_month, model_to_str(model), index, test_mae, weight, next_mae)
            weights.append(weight)
            actuals.append(actuals_d[test_month][model])
            predictions.append(predictions_d[test_month][model])
        check_actuals(actuals)
        # determine ensemble predictions (first k) and regret vs. best model
        ensemble_actuals = actuals[0]  # they are all the same, so pick one
        ensemble_predictions = make_ensemble_predictions(predictions, weights)
        ensemble_rmse, ensemble_mae = errors(ensemble_actuals, ensemble_predictions)
        e.ensemble_detail(ensemble_mae)
        f.ensemble_detail(test_month, ensemble_mae)
        # pick the best models in test_month + 1

        def find_test_month_index(model, model_list):
            index = 0
            while True:
                if model == model_list[index]:
                    return index
                index += 1
            return None  # should not get here

        next_month_mae = mae_d[next_month]
        models_sorted_next_month = sorted(next_month_mae, key=next_month_mae.get)
        first_best_model = models_sorted_next_month[0]
        best_mae = mae_d[next_month][first_best_model]
        for next_month_index in xrange(k):
            a_best_model = models_sorted_next_month[next_month_index]
            test_month_index = find_test_month_index(a_best_model, models_sorted_test_month)
            e.model_best(
                test_month,
                model_to_str(a_best_model),
                test_month_index,
                mae_d[test_month][a_best_model],
                mae_d[next_month][a_best_model],
            )
        # determine regret
        regret = ensemble_mae - best_mae
        e.regret(regret)
        f.regret(test_month, regret)
        median_price = median_prices[next_month]
        e.median_price(median_price)
        f.median_price(test_month, median_price)
        relative_regret = regret / median_price
        e.relative_error(relative_regret)
        f.relative_error(test_month, relative_regret)
        relative_errors.append(relative_regret)
        e.append(' ')
        e.write(control.path_out_e % (k, test_month))
    e.append(' ')
    f.append(' ')
    median_absolute_relative_regret = np.median(np.abs(relative_errors))
    e.marr(median_absolute_relative_regret)  # this line is never seen, because we write earlier
    f.marr(median_absolute_relative_regret)
    f.write(control.path_out_f % k)
    return median_absolute_relative_regret


class ChartGReport():
    def __init__(self, valavm):
        self.report = Report()
        self.format_header = '%4s %7s'
        self.format_detail = '%4d %6.3f%%'
        self._header(valavm)

    def detail(self, k, marr):
        self.report.append(
            self.format_detail % (k, marr * 100.0)
        )

    def _header(self, valavm):
        self.report.append('Hyperparameter K')
        self.report.append(' ')
        self.report.append('Hyperparameter set: %s' % 'all' if valavm == 'roy' else valavm)
        self.report.append(' ')
        self.report.append(
            self.format_header % ('K', 'MARR')
        )

    def write(self, path):
        self.report.append('Legend:')
        self.report.append('K: number of models in ensemble')
        self.report.append('MARR: Median Absolute Relative Regret')
        self.report.write(path)

    def append(self, line):
        self.report.append(line)


def make_charts_efg(actuals_d, predictions_d, mae_d, control, median_prices):
    g = ChartGReport(control.arg.valavm)
    ks = range(1, 31, 1)
    ks.extend([40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200])
    for k in ks:
        median_absolute_relative_regret = make_charts_ef(
            k,
            actuals_d, predictions_d, mae_d, control, median_prices,
        )
        g.detail(k, median_absolute_relative_regret)
    g.write(control.path_out_g)


def make_charts_OLD(reduction_df, actuals_d, predictions_d, mae_d, control, median_prices):
    print 'making charts'

    make_chart_a(reduction_df, control)
    make_chart_b(reduction_df, control)

    def make_sorted_hps(reduction):
        'return dict; key = test_month; value = df of HPs, sorted by MAE'
        def subset_sorted(test_month):
            mask = reduction.test_month == str(test_month)
            return reduction.loc[mask].sort_values('mae')
        return {test_month: subset_sorted(test_month)
                for test_month in test_months()
                }

    sorted_hps = make_sorted_hps(reduction_df)

    def detail_lines_c(sorted_hps_test_month):
        return [0]  # the one with the lowest MAE (the best set of HPs)

    make_chart_cd(reduction_df, control, median_prices, sorted_hps, detail_lines_c, 'c')
    for n_best in (5, 100):
        report_id = 'd-%0d' % n_best

        def detail_lines_d(sorted_hps_test_month):
            result = []
            for k in xrange(len(sorted_hps_test_month)):
                result.append(k)
                if k == (n_best - 1):
                    break
            result.append(len(sorted_hps_test_month) - 1)
            return result

        make_chart_cd(reduction_df, control, median_prices, sorted_hps, detail_lines_d, report_id)
    make_charts_efg(actuals_d, predictions_d, mae_d, control, median_prices)


def make_charts(avm_results, market_results, control):
    print 'making charts'

    make_chart_a(avm_results, control)
    make_chart_b(avm_results, control)

    def detail_lines_c(sorted_hps_test_month):
        return [0]  # the one with the lowest MAE (the best set of HPs)

    make_chart_cd(avm_results, market_results, control, detail_lines_c, 'c')
    for n_best in (5, 100):
        report_id = 'd-%0d' % n_best

        def detail_lines_d(sorted_hps_test_month):
            result = []
            for k in xrange(len(sorted_hps_test_month)):
                result.append(k)
                if k == (n_best - 1):
                    break
            result.append(len(sorted_hps_test_month) - 1)
            return result

        make_chart_cd(avm_results, market_results, control, detail_lines_d, report_id)
    make_charts_efg(avm_results, market_results, control)


def errors(actuals, predictions):
    'return root_mean_squared_error, median_absolute_error'
    errors = actuals - predictions
    root_mean_squared_error = np.sqrt(np.sum(errors * errors) / (1.0 * len(errors)))
    median_absolute_error = np.median(np.abs(errors))
    return root_mean_squared_error, median_absolute_error


def extract_yyyymm(path):
    return path.split('/')[4].split('.')[0]


def make_ci95(v):
    'return tuple with 95 percent confidence interval for the value in np.array v'
    n_samples = 10000
    samples = np.random.choice(v, size=n_samples, replace=True)  # draw with replacement
    sorted_samples = np.sort(samples)
    ci = (sorted_samples[int(n_samples * 0.025) - 1], sorted_samples[int(n_samples * 0.975) - 1])
    return ci


ReductionKey = collections.namedtuple(
    'ReductionKey',
    'model validation_year validation_month n_months_back hps'
)
ReductionValue = collections.namedtuple(
    'ReductionValue',
    'ci95 mae'
)


class Reduction(collections.MutableMapping):  # use Abstract Base Class mixin
    'emulate a dict container with added selection functions'
    def __init__(self):
        self._d = {}

    def __setitem__(self, key, value):
        'key:(year_month, avm_reduction_key)  value:avm_value'
        validation_year_month, validation_key = key

        # build key
        model = ('en' if isinstance(validation_key, ResultKeyEn) else
                 'rf' if isinstance(validation_key, ResultKeyRfr) else
                 'gb'
                 )
        validation_year = validation_year_month // 100
        validation_month = validation_year_month - 100 * validation_year
        n_months_back = validation_key.n_months_back
        # del validation_key['n_months_back']
        # hps = tuple(sorted(validation_key))  # re-create with dict(hps)
        key = ReductionKey(
            model=model,
            validation_year=validation_year,
            validation_month=validation_month,
            n_months_back=n_months_back,
            hps=validation_key,
        )

        # build value
        actuals = np.array(value.actuals)
        predictions = np.array(value.predictions)
        errors = actuals - predictions
        mae = np.median(np.abs(errors))
        ci95 = make_ci95(errors)
        value = ReductionValue(
            ci95=ci95,
            mae=mae,
        )

        self._d[key] = value

    def __getitem__(self, key):
        return self._d[key]

    def __delitem__(self, key):
        del self._d[key]

    def __len__(self):
        return len(self._d)

    def __str__(self):
        return 'Reduction(len=%d)' % len(self._d)

    def __iter__(self):
        return self._d.__iter__()

    def update(self, other):
        return self._d.update(other)

    def select_maes_where_model_year_month(self, model, year, month):
        'return list of median absolute errors for specified model, year, and month'
        pdb.set_trace()
        result = []
        for key, value in self._d.iteritems():
            if (key.model == model) and (key.validation_year == year) and (key.validation_month == month):
                result.append(value.mae)
        pdb.set_trace()
        return result


def make_data(control):
    'return a Reduction containing all of the data'
    def reduce_path(path):
        'return a Reduction containing the data for one validation month'
        debug = False
        reduction = Reduction()
        validation_year_month = int(extract_yyyymm(path))
        n_read = 0
        with open(path, 'rb') as f:
            while True:
                if debug and (n_read > 5):
                    break
                try:
                    validation_record = pickle.load(f)  # read until EOF
                    n_read += 1
                    assert isinstance(validation_record, tuple), type(validation_record)
                    validation_key, validation_value = validation_record
                    reduction_key = (validation_year_month, validation_key)
                    if reduction_key in reduction:
                        print 'duplicate key: should not happen'
                        print reduction_key
                        pdb.set_trace()
                    reduction[(validation_year_month, validation_key)] = validation_value
                except ValueError as e:
                    print 'ValueError', e  # ignore error
                    if validation_key is not None:
                        print validation_key
                    if validation_value is not None:
                        print validation_value
                    print 'ValueError %s on record %d ignored; record discarded' % (e, n_read + 1)
                except EOFError:
                    break
        return reduction

    reduction = Reduction()
    paths = sorted(glob.glob(control.path_in_ege))
    for path in paths:
        print 'reducing', path
        path_reduction = reduce_path(path)
        print 'created %d records from path %s' % (len(path_reduction), path)
        reduction.update(path_reduction)
    print 'create %d records from all paths' % len(reduction)
    return reduction


def make_data_old(control):
    'return reduction data frame, reduction dict, ege_control'

    actuals_d = collections.defaultdict(dict)
    predictions_d = collections.defaultdict(dict)
    mae_d = collections.defaultdict(dict)

    def make_row(test_month, k, v):
        'return a dict, that will be the next row of dataframe in the form of a dict'
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
        mae_d[int(test_month)][k] = mae
        actuals_d[int(test_month)][k] = actuals
        predictions_d[int(test_month)][k] = predictions
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

    def process_records(path, rows_list):
        'mutate rows_list, a list of dictionaries, to include objects at path'
        pdb.set_trace()
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
        process_records(path, rows_list)
    return pd.DataFrame(rows_list), actuals_d, predictions_d, mae_d


class Market(object):
    def __init__(self, chart_01_reduction):
        self._r = chart_01_reduction


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(logfile_path=control.path_out_log)
    print control

#    if control.arg.data:
#        df, actuals_d, predictions_d, mae_d = make_data(control)
#        with open(control.path_data, 'wb') as f:
#            pickle.dump((df, actuals_d, predictions_d, mae_d, control), f)
#        print 'wrote reduction data file'
#    else:
#        with open(control.path_data, 'rb') as f:
#            reduction_df, actuals_d, predictions_d, mae_d, reduction_control = pickle.load(f)
#            with open(control.path_in_median_prices, 'rb') as g:
#                data, reduction_control = pickle.load(g)
#                counts, means, medians, prices = data
#                make_charts(reduction_df, actuals_d, predictions_d, mae_d, control, medians)
    if control.arg.data:
        d = make_data(control)
        with open(control.path_data, 'wb') as f:
            pickle.dump((d, control), f)
        print 'wrote reduction data file'
    else:
        with open(control.path_data, 'rb') as f:
            chart_06_reduction, reduction_control = pickle.load(f)
            with open(control.path_in_chart_01_reduction, 'rb') as g:
                chart_01_data, reduction_control = pickle.load(g)
                # TODO: already load trading volumes (from chart01)
                make_charts(chart_06_reduction, Market(chart_01_data), control)
                # make_charts(reduction_df, actuals_d, predictions_d, mae_d, control, medians)

    print control
    if control.test:
        print 'DISCARD OUTPUT: test'
    if control.debug:
        print 'DISCARD OUTPUT: debug'
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
