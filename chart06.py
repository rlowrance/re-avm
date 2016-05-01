'''create charts showing results of valgbr.py

INVOCATION
  python chart06.py [--data] [--test]
where
  VALAVM in {roy, anil}

INPUT FILES
 WORKING/chart01/data.pickle
 WORKING/samples-train.csv
 WORKING/valavm/YYYYMM.pickle
 WORKING/data.pickle or WORKING/data-test.pickle

INPUT AND OUTPUT FILES (build with --data)
 WORKING/chart06/data.pickle         | reduced data
 WORKING/chart06/data-test.pickle    | reduced data test subset; always built, sometimes read

OUTPUT FILES
 WORKING/chart06/a.pdf          | range of losses by model (graph)
 WORKING/chart06/b-YYYYMM.txt   | HPs with lowest losses
 WORKING/chart06/c.pdf          | best model each month
 WORKING/chart06/d.pdf          | best & 50th best each month
 WORKING/chart06/e.pdf          | best 50 models each month (was chart07)
 WORKING/chart06/best.pickle    | dataframe with best choices each month # CHECK
'''

from __future__ import division

import argparse
import collections
import cPickle as pickle
import glob
import itertools
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
import chart06columns
from ColumnsTable import ColumnsTable
from columns_contain import columns_contain
from Logger import Logger
from Month import Month
from Path import Path
from Report import Report
from Timer import Timer
from valavm import ResultKeyEn, ResultKeyGbr, ResultKeyRfr, ResultValue
cc = columns_contain


ModelDescription = collections.namedtuple(
    'ModelDescription',
    'model n_months_back units_X units_y alpha l1_ratio ' +
    'n_estimators max_features max_depth loss learning_rate'
)


ModelResults = collections.namedtuple(
    'ModelResults',
    'rmse mae ci95_low ci95_high predictions'
)


class ColumnDefinitions(object):
    'all reports use these column definitions'
    def __init__(self):
        self._defs = {
            'median_absolute_error': [6, '%6d', (' ', 'MAE'), 'median absolute error'],
            'model': [5, '%5s', (' ', 'model'),
                      'model name (en = elastic net, gd = gradient boosting, rf = random forests)'],
            'n_months_back': [2, '%2d', (' ', 'bk'), 'number of mnths back for training'],
            'max_depth': [4, '%4d', (' ', 'mxd'), 'max depth of any individual decision tree'],
            'n_estimators': [4, '%4d', (' ', 'next'), 'number of estimators (= number of trees)'],
            'max_features': [4, '%4s', (' ', 'mxft'), 'maximum number of features examined to split a node'],
            'learning_rate': [4, '%4.1f', (' ', 'lr'), 'learning rate for gradient boosting'],
            'alpha': [5, '%5.2f', (' ', 'alpha'), 'constant multiplying penalty term for elastic net'],
            'l1_ratio': [4, '%4.2f', (' ', 'l1'), 'l1_ratio mixing L1 and L2 penalties for elastic net'],
            'units_X': [6, '%6s', (' ', 'unitsX'), 'units for the x value; either natural (nat) or log'],
            'units_y': [6, '%6s', (' ', 'unitsY'), 'units for the y value; either natural (nat) or log'],
            'validation_month': [6, '%6s', ('vald', 'month'), 'month used for validation'],
            'rank': [4, '%4d', (' ', 'rank'), 'rank within validation month; 1 == lowest MAE'],
            'median_price': [6, '%6d', ('median', 'price'), 'median price in the validation month'],
            'mae_validation': [6, '%6d', ('vald', 'MAE'), 'median absolute error in validation month'],
            'mae_next': [6, '%6d', ('next', 'MAE'),
                         'median absolute error in test month (which follows the validation month)'],
            'note': [15, '%15s', (' ', 'note'),
                     'when provided, the next MAE column contains the specified value'],
            'rank_index': [5, '%5d', ('rank', 'index'),
                           'ranking of model performance in the validation month; 0 == best'],
            'weight': [6, '%6.4f', (' ', 'weight'), 'weight of the model in the ensemble method'],
            'ensemble_mae': [6, '%6d', ('ensb', 'MAE'),
                             'median absolute error of the ensemble model'],
            'best_next_month_mae': [6, '%6d', ('best', 'MAE'),
                                    'median absolute error of the best model in the next month']

        }

    def defs_for_columns(self, *key_list):
        return [[key] + self._defs[key]
                for key in key_list
                ]

    def replace_by_spaces(self, k, v):
        'define values that are replaced by spaces'
        if isinstance(v, float) and np.isnan(v):
            return True
    return False


def make_control(argv):
    # return a Bunch

    print argv
    parser = argparse.ArgumentParser()
    parser.add_argument('invocation')
    parser.add_argument('--data', action='store_true')
    parser.add_argument('--test', action='store_true')
    arg = parser.parse_args(argv)  # arg.__dict__ contains the bindings
    arg.base_name = 'chart06'

    random_seed = 123
    random.seed(random_seed)

    # assure output directory exists
    dir_working = Path().dir_working()
    dir_out = dir_working + 'chart06/'
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    return Bunch(
        arg=arg,
        column_definitions=ColumnDefinitions(),
        debug=False,
        path_in_ege=(
            dir_working +
            'valavm' +
            '/??????.pickle'),
        path_in_samples=dir_working + 'samples-train.csv',
        path_in_data=dir_out + ('data-test.pickle' if arg.test else 'data.pickle'),
        path_out_a=dir_out + 'a.pdf',
        path_out_b=dir_out + 'b-%d.txt',
        path_out_cd=dir_out + '%s.txt',
        path_out_d=dir_out + 'd.txt',
        path_out_e=dir_out + 'e-%04d-%6d.txt',
        path_out_f=dir_out + 'f-%04d.txt',
        path_out_g=dir_out + 'g.txt',
        path_out_data=dir_out + 'data.pickle',
        path_out_data_test=dir_out + 'data-test.pickle',
        path_out_log=dir_out + 'log' + ('-data' if arg.data else '') + '.txt',
        path_in_chart_01_reduction=dir_working + 'chart01/data.pickle',
        random_seed=random_seed,
        sampling_rate=0.02,
        test=arg.test,
        timer=Timer(),
        validation_months=('200612', '200701', '200702', '200703', '200704', '200705',
                           '200706', '200707', '200708', '200709', '200710', '200711',
                           ),
    )


def select_and_sort(df, year, month, model):
    'return new df contain sorted observations for specified year, month, model'
    yyyymm = str(year * 100 + month)
    mask = (
        (df.model == model) &
        (df.validation_month == yyyymm)
    )
    subset = df.loc[mask]
    if len(subset) == 0:
        print 'empty subset'
        print year, month, model, sum(df.model == model), sum(df.validation_month == yyyymm)
        pdb.set_trace()
    return subset.sort_values('mae')


def make_chart_a(reduction, control):
    'graph range of errors by month by method'
    print 'make_chart_a'

    def make_subplot(validation_month):
        'mutate the default axes'
        # draw one line for each model family
        for model in ('en', 'gb', 'rf'):
            y = [v.mae
                 for k, v in reduction[validation_month].iteritems()
                 if k.model == model
                 ]
            plt.plot(y, label=model)  # the reduction is sorted by increasing mae
            plt.yticks(size='xx-small')
            plt.title('%s' % (validation_month),
                      loc='right',
                      fontdict={'fontsize': 'xx-small',
                                'style': 'italic',
                                },
                      )
            plt.xticks([])  # no ticks on x axis
        return

    def make_figure():
        # make and save figure

        plt.figure()  # new figure
        # plt.suptitle('Loss by Test Period, Tree Max Depth, N Trees')  # overlays the subplots
        axes_number = 0
        # MAYBE: add first 6 months in 2008
        validation_months = ('200612', '200701', '200702', '200703', '200704', '200705',
                             '200706', '200707', '200708', '200709', '200710', '200711',
                             )
        row_seq = (1, 2, 3, 4)
        col_seq = (1, 2, 3)
        for row in row_seq:
            for col in col_seq:
                validation_month = validation_months[axes_number]
                axes_number += 1  # count across rows
                plt.subplot(len(row_seq), len(col_seq), axes_number)
                make_subplot(validation_month)
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


def differ_only_in_unitsOLD(a, b):
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


def differOLD(a, b):
    if a is None:
        return b is None
    for index, value in a.iteritems():
        if isinstance(value, float) and np.isnan(value) and (not np.isnan(b[index])):
            return True
        if value != b[index]:
            return True
    return False


def make_hp_stringOLD(series):
    ignored_indices = set(('mae', 'model', 'yyyymm', 'n_months_back',
                           'max_depth', 'n_estimators', 'max_features',
                           'validation_month', 'loss', 'rmse',
                           'ci95_low', 'ci95_high'))
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


def nan_to_NoneOLD(x):
    print x, type(x)
    if x is None:
        return None
    return None if np.isnan(x) else x


class ChartBReport(object):
    def __init__(self, validation_month, k):
        self._report = Report()
        self._header(validation_month, k)
        cd = chart06columns.defs_for_columns(
            'median_absolute_error', 'model', 'n_months_back',
            'max_depth', 'n_estimators', 'max_features',
            'learning_rate',
        )
        self._ct = ColumnsTable(columns=cd, verbose=True)

    def _header(self, validation_month, k):
        def a(line):
            self._report.append(line)

        a('MAE for %d best-performing models and their hyperparameters' % k)
        a('Validation month: %s' % validation_month)
        a(' ')

    def append_detail(self, **kwds):
        # replace NaN with None
        with_spaces = {k: (None if chart06columns.replace_by_spaces(k, v) else v)
                       for k, v in kwds.iteritems()
                       }
        self._ct.append_detail(**with_spaces)

    def write(self, path):
        self._ct.append_legend()
        for line in self._ct.iterlines():
            self._report.append(line)
        self._report.write(path)


def make_chart_b(reduction, control):
    def write_report(year, month):
        validation_month = str(year * 100 + month)
        k = 50  # report on the first k models in the sorted subset
        report = ChartBReport(validation_month, k)
        detail_line_number = 0
        # ref: http://stackoverflow.com/questions/7971618/python-return-first-n-keyvalue-pairs-from-dict
        first_k = itertools.islice(reduction[validation_month].items(), 0, k)
        for key, value in first_k:
            report.append_detail(
                median_absolute_error=value.mae,
                model=key.model,
                n_months_back=key.n_months_back,
                max_depth=key.max_depth,
                n_estimators=key.n_estimators,
                max_features=key.max_features,
                learning_rate=key.learning_rate,
            )
            detail_line_number += 1
            if detail_line_number > k:
                break
        report.write(control.path_out_b % int(validation_month))

    for year in (2006, 2007):
        months = (12,) if year == 2006 else (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
        for month in months:
            write_report(year, month)


class ChartCDReport(object):
    def __init__(self):
        self._report = Report()
        cd = chart06columns.defs_for_columns(
            'validation_month', 'rank', 'median_absolute_error',
            'median_price', 'model', 'n_months_back',
            'max_depth', 'n_estimators', 'max_features',
            'learning_rate', 'alpha', 'l1_ratio',
            'units_X', 'units_y',
        )
        self._ct = ColumnsTable(columns=cd, verbose=True)
        self._header()

    def append(self, line):
        self._report.append(line)

    def write(self, path):
        self._ct.append_legend()
        for line in self._ct.iterlines():
            self._report.append(line)
        self._report.write(path)

    def _header(self):
        self._report.append('Median Absolute Error (MAE) by month for best-performing models and their hyperparameters')
        self._report.append(' ')

    def append_detail(self, **kwds):
        with_spaces = {k: (None if chart06columns.replace_by_spaces(k, v) else v)
                       for k, v in kwds.iteritems()
                       }
        self._ct.append_detail(**with_spaces)


def make_chart_cd(reduction, median_prices, control, detail_line_indices, report_id):
    r = ChartCDReport()
    for validation_month in control.validation_months:
        median_price = median_prices[validation_month]
        month_result_keys = list(reduction[validation_month].keys())
        for detail_line_index in detail_line_indices:
            k = month_result_keys[detail_line_index]
            v = reduction[validation_month][k]
            r.append_detail(
                validation_month=validation_month,
                rank=detail_line_index + 1,
                median_absolute_error=v.mae,
                median_price=median_price,
                model=k.model,
                n_months_back=k.n_months_back,
                max_depth=k.max_depth,
                n_estimators=k.n_estimators,
                max_features=k.max_features,
                learning_rate=k.learning_rate,
                alpha=k.alpha,
                l1_ratio=k.l1_ratio,
                units_X=k.units_X[:3],
                units_y=k.units_y[:3],
            )

    r.write(control.path_out_cd % report_id)
    return


class ChartEReportOLD(object):
    def __init__(self, k, ensemble_weighting):
        self.report = Report()
        self.format_header = '%6s %20s %5s %6s %6s %6s'
        self.format_model = '%6d %20s %5d %6.0f %6.4f %6.0f'
        self.format_best = '%6d %20s %5d %6.0f %6s %6.0f best'
        self.format_ensemble = '%6s %20s %5s %6s %6s %6.0f'
        self.format_relative_error = '%6s %20s %5s %6s %6s %5.3f%%'
        self.format_marr = '%6s %20s %5s %6s %6s %6.2%%'
        self._header(k, ensemble_weighting)

    def _header(self, k, ensemble_weighting):
        self.report.append('Performance of Best Models Separately and as an Ensemble')
        self.report.append(' ')
        self.report.append('Consider best K = %d models' % k)
        self.report.append('Ensemble weighting: %s' % ensemble_weighting)

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


class ChartEReport(object):
    'has 2 kinds of detail lines'
    def __init__(self, k, ensemble_weighting):
        self._report = Report()
        cd = chart06columns.defs_for_columns(
            'validation_month', 'model', 'n_months_back',
            'n_estimators', 'max_features', 'max_depth',
            'learning_rate', 'rank', 'mae_validation',
            'weight', 'mae_next', 'note'
        )
        self._ct = ColumnsTable(columns=cd, verbose=True)
        self._header(k, ensemble_weighting)

    def write(self, path):
        self._ct.append_legend()
        for line in self._ct.iterlines():
            self._report.append(line)
        self._report.write(path)

    def detail_line(self, **kwds):
        with_spaces = {k: (None if chart06columns.replace_by_spaces(k, v) else v)
                       for k, v in kwds.iteritems()
                       }
        self._ct.append_detail(**with_spaces)

    def detail_note(self, value, text):
        self._ct.append_detail(**{'mae_next': value, 'note': text})

    def _header(self, k, ensemble_weighting):
        self._report.append('Performance of Best Models Separately and as an Ensemble')
        self._report.append(' ')
        self._report.append('Considering Best K = %d models' % k)
        self._report.append('Ensemble weighting: %s' % ensemble_weighting)


class ChartFReport(object):  # OLD API
    def __init__(self, k, ensemble_weighting):
        self.report = Report()
        self.format_header = '%6s %20s %6s'
        self.format_ensemble = '%6d %20s  %6.0f'
        self.format_relative_error = '%6d %20s %5.3f%%'
        self.format_regret = '%6d %20s %6.0f'
        self.format_marr = '%6s %20s %5.3f%%'
        self._header(k, ensemble_weighting)

    def _header(self, k, ensemble_weighting):
        self.report.append('Regret of Ensemble Models')
        self.report.append(' ')
        self.report.append('Consider best %d models' % k)
        self.report.append('Ensemble weighting: %s' % ensemble_weighting)
        self.report.append(' ')
        self.report.append(
            self.format_header % (
                'vald.', ' ', 'next'))
        self.report.append(
            self.format_header % (
                'month', 'model', 'MAE'))

    def ensemble_detail(self, validation_month, mae):
        self.report.append(
            self.format_ensemble % (
                validation_month, 'ensemble', mae))

    def median_price(self, validation_month, amount):
        self.report.append(
            self.format_ensemble % (
                validation_month, 'median price', amount))

    def regret(self, validation_month, amount):
        self.report.append(
            self.format_regret % (
                validation_month, 'regret', amount))

    def relative_error(self, validation_month, amount):
        self.report.append(
            self.format_relative_error % (
                validation_month, 'relative regret', amount * 100.0))

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


def model_to_strOLD(k):
    pdb.set_trace()  # replace me
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


def make_charts_efOLD(k, reduction, median_price, control):
    pass
#    '''Write charts e and f, return median-absolute-relative_regret object'''
#    verbose = True
#    if verbose:
#        pdb.set_trace()
#    ensemble_weighting = 'exp(-MAE/100000)'
#    relative_errors = []
#    f = ChartFReport(k, ensemble_weighting)
#    for validation_month in control.validation_months:
#        e = ChartEReport(k, ensemble_weighting)
#        validation_month_mae = mae_d[validation_month]
#        models_sorted_validation_month = sorted(
#            validation_month_mae,
#            key=validation_month_mae.get)  # keys in order of mae
#        weights = []
#        actuals = []
#        predictions = []
#        next_month = Month(validation_month).increment(1).as_int()
#        for index in xrange(k):
#            model = models_sorted_validation_month[index]
#            test_mae = mae_d[validation_month][model]
#            next_mae = mae_d[next_month][model]
#            weight = math.exp(-test_mae / 100000.0)
#            e.model_detail(validation_month, model_to_str(model), index, test_mae, weight, next_mae)
#            weights.append(weight)
#            actuals.append(actuals_d[validation_month][model])
#            predictions.append(predictions_d[validation_month][model])
#        check_actuals(actuals)
#        # determine ensemble predictions (first k) and regret vs. best model
#        ensemble_actuals = actuals[0]  # they are all the same, so pick one
#        ensemble_predictions = make_ensemble_predictions(predictions, weights)
#        ensemble_rmse, ensemble_mae, ensemble_ci95_low, ensemble_ci95_high = errors(
#            ensemble_actuals,
#            ensemble_predictions,
#        )
#        e.ensemble_detail(ensemble_mae)
#        f.ensemble_detail(validation_month, ensemble_mae)
#        # pick the best models in validation_month + 1
#
#        def find_validation_month_index(model, model_list):
#            index = 0
#            while True:
#                if model == model_list[index]:
#                    return index
#                index += 1
#            return None  # should not get here
#
#        next_month_mae = mae_d[next_month]
#        models_sorted_next_month = sorted(next_month_mae, key=next_month_mae.get)
#        first_best_model = models_sorted_next_month[0]
#        best_mae = mae_d[next_month][first_best_model]
#        for next_month_index in xrange(k):
#            a_best_model = models_sorted_next_month[next_month_index]
#            validation_month_index = find_validation_month_index(a_best_model, models_sorted_validation_month)
#            e.model_best(
#                validation_month,
#                model_to_str(a_best_model),
#                validation_month_index,
#                mae_d[validation_month][a_best_model],
#                mae_d[next_month][a_best_model],
#            )
#        # determine regret
#        regret = ensemble_mae - best_mae
#        e.regret(regret)
#        f.regret(validation_month, regret)
#        median_price = time_period_stats[next_month]['median']
#        e.median_price(median_price)
#        f.median_price(validation_month, median_price)
#        relative_regret = regret / median_price
#        e.relative_error(relative_regret)
#        f.relative_error(validation_month, relative_regret)
#        relative_errors.append(relative_regret)
#        e.append(' ')
#        e.write(control.path_out_e % (k, validation_month))
#    e.append(' ')
#    f.append(' ')
#    median_absolute_relative_regret = np.median(np.abs(relative_errors))
#    e.marr(median_absolute_relative_regret)  # this line is never seen, because we write earlier
#    f.marr(median_absolute_relative_regret)
#    f.write(control.path_out_f % k)
#    return median_absolute_relative_regret
#


def make_charts_ef(k, reduction, actuals, median_price, control):
    '''Write charts e and f, return median-absolute-relative_regret object'''
    verbose = True
    if verbose:
        pdb.set_trace()
    ensemble_weighting = 'exp(-MAE/100000)'
    relative_errors = []
    f = ChartFReport(k, ensemble_weighting)
    for validation_month in control.validation_months:
        e = ChartEReport(k, ensemble_weighting)
        next_month = Month(validation_month).increment(1).as_str()
        validation_month_keys = list(reduction[validation_month].keys())
        cum_weighted_predictions = None
        cum_weights = 0
        for index in xrange(k):
            validation_month_key = validation_month_keys[index]
            validation_month_value = reduction[validation_month][k]
            next_month_value = reduction[next_month][validation_month_key]
            mae_validation = validation_month_value.mae
            mae_next = next_month_value.mae
            weight = math.exp(-mae_validation / 100000.0)
            e.detail_line(
                validation_month=validation_month,
                model=validation_month_key.model,
                n_months_back=validation_month_key.n_months_back,
                n_estimators=validation_month_key.n_estimators,
                max_features=validation_month_key.max_features,
                max_depth=validation_month_key.max_depth,
                learning_rate=validation_month_key.learning_rate,
                rank=index + 1,
                mae_validation=mae_validation,
                weight=weight,
                mae_next=mae_next,
                note=' ',
            )
            # need the mae of the ensemble
            # need the actuals and predictions? or is this already computed
            predictions_next = next_month_value.predictions
            if cum_weighted_predictions is None:
                cum_weighted_predictions = weight * predictions_next
            else:
                cum_weighted_predictions += weight * predictions_next
            cum_weights += weight
        pdb.set_trace()
        ensemble_predictions = cum_weighted_predictions / cum_weights
        ensemble_rmse, ensemble_mae, ensemble_ci95_low, ensemble_ci95_high = errors(
            actuals[next_month],
            ensemble_predictions,
        )
        best_key = reduction[next_month].keys()[0]
        best_value = reduction[next_month][best_key]
        e.detail_line(
            validation_month=validation_month,
            ensemble_mae=ensemble_mae,
            best_next_month_mae=best_value.mae,
        )
        e.write(control.path_out_e % (k, validation_month))
    return  # for now, we may need to return a value later
    e.append(' ')
    f.append(' ')
    median_absolute_relative_regret = np.median(np.abs(relative_errors))
    e.marr(median_absolute_relative_regret)  # this line is never seen, because we write earlier
    f.marr(median_absolute_relative_regret)
    f.write(control.path_out_f % k)
    return median_absolute_relative_regret


class ChartGReport():
    def __init__(self):
        self.report = Report()
        self.format_header = '%4s %7s'
        self.format_detail = '%4d %6.3f%%'
        self._header()

    def detail(self, k, marr):
        self.report.append(
            self.format_detail % (k, marr * 100.0)
        )

    def _header(self):
        self.report.append('Hyperparameter K')
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


def make_charts_efg(reduction, actuals, median_prices, control):
    # chart g uses the regret values that are computed in building chart e
    g = ChartGReport()
    ks = range(1, 31, 1)
    ks.extend([40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200])
    for k in ks:
        median_absolute_relative_regret = make_charts_ef(k, reduction, actuals, median_prices, control)
        g.detail(k, median_absolute_relative_regret)
    g.write(control.path_out_g)


def make_charts(reduction, actuals, median_prices, control):
    print 'making charts'

    make_chart_a(reduction, control)
    make_chart_b(reduction, control)

    make_chart_cd(reduction, median_prices, control, (0,), 'c')
    for n_best in (5, 100):
        report_id = 'd-%0d' % n_best
        for validation_month, month_reduction in reduction.iteritems():
            n_reductions_per_month = len(month_reduction)
            break
        detail_lines_d = range(n_best)[:n_reductions_per_month]
        make_chart_cd(reduction, median_prices, control, detail_lines_d, report_id)
    make_charts_efg(reduction, actuals, median_prices, control)


def errors(actuals, predictions):
    'return root_mean_squared_error, median_absolute_error'
    def make_ci95(v):
        'return tuple with 95 percent confidence interval for the value in np.array v'
        n_samples = 10000
        samples = np.random.choice(v, size=n_samples, replace=True)  # draw with replacement
        sorted_samples = np.sort(samples)
        ci = (sorted_samples[int(n_samples * 0.025) - 1], sorted_samples[int(n_samples * 0.975) - 1])
        return ci

    errors = actuals - predictions
    mse = np.sum(errors * errors) / len(errors)
    root_mean_squared_error = np.sqrt(mse)
    median_absolute_error = np.median(np.abs(errors))
    ci95_low, ci95_high = make_ci95(errors)
    return root_mean_squared_error, median_absolute_error, ci95_low, ci95_high


def extract_yyyymm(path):
    return path.split('/')[4].split('.')[0]


def make_data(control):
    'return dict[validation_month][ModelDescription] = ModelResults, sorted by increasing MAE'

    def append(validation_month, k, v, reduction, actuals_by_validation_month):
        'add entry to result'
        is_en = isinstance(k, ResultKeyEn)
        is_gbr = isinstance(k, ResultKeyGbr)
        is_rfr = isinstance(k, ResultKeyRfr)
        is_tree = is_gbr or is_rfr
        model = 'en' if is_en else ('gb' if is_gbr else 'rf')
        actuals = v.actuals.values
        predictions = v.predictions
        if validation_month in actuals_by_validation_month:
            if (actuals == actuals_by_validation_month[validation_month]).all():
                pass
            else:
                print '++++++++++++++++++++'
                print validation_month
                print actuals
                print actuals_by_validation_month[validation_month]
                print 'should be the same'
                print '++++++++++++++++++++'
                pdb.set_trace()
        else:
            actuals_by_validation_month[validation_month] = actuals
        if predictions is None:
            print '+++++++++++++++++++'
            print k
            print 'predictions is missing'
            print '+++++++++++++++++++'
            pdb.set_trace()
        rmse, mae, ci95_low, ci95_high = errors(actuals, predictions)
        key = ModelDescription(
            model=model,
            n_months_back=k.n_months_back,
            units_X=k.units_X if is_en else 'natural',
            units_y=k.units_y if is_en else 'natural',
            alpha=k.alpha if is_en else None,
            l1_ratio=k.l1_ratio if is_en else None,
            n_estimators=k.n_estimators if is_tree else None,
            max_features=k.max_features if is_tree else None,
            max_depth=k.max_depth if is_tree else None,
            loss=k.loss if is_gbr else None,
            learning_rate=k.learning_rate if is_gbr else None,
        )
        value = ModelResults(
            rmse=rmse,
            mae=mae,
            ci95_low=ci95_low,
            ci95_high=ci95_high,
            predictions=predictions,
        )
        if key in reduction[validation_month]:
            print '++++++++++++++++++++'
            print validation_month
            print key
            print 'duplicate key in validation month'
            print '++++++++++++++++++++'
            pdb.set_trace()
        reduction[validation_month][key] = value
        return

    def process_records(path, reduction, all_actuals):
        'mutate reduction, a dict, to include objects at path'
        print 'reducing', path
        validation_month = extract_yyyymm(path)
        n = 0
        with open(path, 'rb') as f:
            while True:
                try:
                    record = pickle.load(f)  # read until EOF
                    assert isinstance(record, tuple), type(record)
                    key, value = record
                    n += 1
                    append(validation_month, key, value, reduction, all_actuals)
                except ValueError as e:
                    if record is not None:
                        print record
                    print 'ignoring ValueError in record %d: %s' % (n, e)
                except EOFError:
                    break
        print 'number of records retained', n

    control.debug = True
    reduction = collections.defaultdict(dict)
    all_actuals = {}
    paths = sorted(glob.glob(control.path_in_ege))
    for path in paths:
        process_records(path, reduction, all_actuals)

    # sort by increasing mae
    ordered = {}
    for validation_month, results in reduction.iteritems():
        sorted_results = collections.OrderedDict(sorted(results.items(), key=lambda t: t[1].mae))
        ordered[validation_month] = sorted_results
    return ordered, all_actuals


class Sampler(object):
    def __init__(self, fraction):
        self._fraction = fraction
        self._keys = None

    def sample(self, d):
        if self._keys is None:
            k = int(len(d) * self._fraction)
            self._keys = random.sample(list(d), k)
        sample = {key: d[key] for key in self._keys}
        return sample


def make_samples(reduction, fraction):
    'return a random sample of the reduction stratified by validation_month'
    # use same keys (models) every validation month
    samples = collections.defaultdict(dict)
    sampler = Sampler(fraction)
    for validation_month, validation_dict in reduction.iteritems():
        selected = sampler.sample(validation_dict)
        for k, v in selected.iteritems():
            samples[validation_month][k] = v
    return samples


def make_median_price(path):
    with open(path, 'rb') as f:
        d, reduction_control = pickle.load(f)
        median_price = {}
        for validation_month, v in d.iteritems():
            median_price[str(validation_month)] = v['median']
        return median_price


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(logfile_path=control.path_out_log)
    print control
    lap = control.timer.lap

    if control.arg.data:
        median_price = make_median_price(control.path_in_chart_01_reduction)
        lap('make_median_price')
        reduction, all_actuals = make_data(control)
        lap('make_data')
        samples = make_samples(reduction, control.sampling_rate)
        lap('make_samples')
        output_all = (reduction, all_actuals, median_price, control)
        output_samples = (samples, all_actuals, median_price, control)
        with open(control.path_out_data, 'wb') as f:
            pickle.dump(output_all, f)
            lap('write all data')
        with open(control.path_out_data_test, 'wb') as f:
            pickle.dump(output_samples, f)
            lap('write samples')
    else:
        with open(control.path_in_data, 'rb') as f:
            print 'reading reduction data file'
            reduction, all_actuals, median_price, reduction_control = pickle.load(f)
            lap('read input from %s' % control.path_in_data)
        make_charts(reduction, all_actuals, median_price, control)

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
