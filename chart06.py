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
            'n_estimators': [4, '%4d', (' ', 'nest'), 'number of estimators (= number of trees)'],
            'max_features': [4, '%4s', (' ', 'mxft'), 'maximum number of features examined to split a node'],
            'learning_rate': [4, '%4.1f', (' ', 'lr'), 'learning rate for gradient boosting'],
            'alpha': [5, '%5.2f', (' ', 'alpha'), 'constant multiplying penalty term for elastic net'],
            'l1_ratio': [4, '%4.2f', (' ', 'l1'), 'l1_ratio mixing L1 and L2 penalties for elastic net'],
            'units_X': [6, '%6s', (' ', 'unitsX'), 'units for the x value; either natural (nat) or log'],
            'units_y': [6, '%6s', (' ', 'unitsY'), 'units for the y value; either natural (nat) or log'],
            'validation_month': [6, '%6s', ('vald', 'month'), 'month used for validation'],
            'rank': [4, '%4d', (' ', 'rank'), 'rank within validation month; 1 == lowest MAE'],
            'median_price': [6, '%6d', ('median', 'price'), 'median price in the validation month'],
            'rank_index': [5, '%5d', ('rank', 'index'),
                           'ranking of model performance in the validation month; 0 == best'],
            'weight': [6, '%6.4f', (' ', 'weight'), 'weight of the model in the ensemble method'],
            'mae_ensemble': [6, '%6d', ('ensb', 'MAE'),
                             'median absolute error of the ensemble model'],
            'mae_best_next_month': [6, '%6d', ('best', 'MAE'),
                                    'median absolute error of the best model in the next month'],
            'mae_next': [6, '%6d', ('next', 'MAE'),
                         'median absolute error in test month (which follows the validation month)'],
            'mae_validation': [6, '%6d', ('vald', 'MAE'), 'median absolute error in validation month'],
            'fraction_median_price_next_month_ensemble': [
                6, '%6.3f', ('ensmbl', 'relerr'),
                'ensemble MAE as a fraction of the median price in the next month'],
            'fraction_median_price_next_month_best': [
                6, '%6.3f', ('best', 'relerr'),
                'best model MAE as a fraction of the median price in the next month'],

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


def trace_unless(condition, message, **kwds):
    if condition:
        return
    print '+++++++++++++++'
    for k, v in kwds.iteritems():
        print k, v
    print message
    print '+++++++++++++++'
    pdb.set_trace()


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
        path_out_e=dir_out + 'e-%04d-%6s.txt',
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
    def __init__(self, validation_month, k, column_definitions):
        self._report = Report()
        self._header(validation_month, k)
        self._column_definitions = column_definitions
        cd = self._column_definitions.defs_for_columns(
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
        with_spaces = {
            k: (None if self._column_definitions.replace_by_spaces(k, v) else v)
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
        report = ChartBReport(validation_month, k, control.column_definitions)
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
    def __init__(self, column_definitions):
        self._column_definitions = column_definitions
        self._report = Report()
        cd = self._column_definitions.defs_for_columns(
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
        with_spaces = {
            k: (None if self._column_definitions.replace_by_spaces(k, v) else v)
            for k, v in kwds.iteritems()
        }
        self._ct.append_detail(**with_spaces)


def make_chart_cd(reduction, median_prices, control, detail_line_indices, report_id):
    r = ChartCDReport(control.column_definitions)
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
    def __init__(self, k, ensemble_weighting, column_definitions):
        self._column_definitions = column_definitions
        self._report = Report()
        self._header(k, ensemble_weighting)
        cd = self._column_definitions.defs_for_columns(
            'validation_month', 'model', 'n_months_back',
            'n_estimators', 'max_features', 'max_depth',
            'learning_rate', 'rank', 'weight',
            'mae_validation', 'mae_next', 'mae_ensemble',
            'mae_best_next_month'
        )
        self._ct = ColumnsTable(columns=cd, verbose=True)

    def write(self, path):
        self._ct.append_legend()
        for line in self._ct.iterlines():
            self._report.append(line)
        self._report.write(path)

    def detail_line(self, **kwds):
        with_spaces = {
            k: (None if self._column_definitions.replace_by_spaces(k, v) else v)
            for k, v in kwds.iteritems()
        }
        self._ct.append_detail(**with_spaces)

    def _header(self, k, ensemble_weighting):
        self._report.append('Performance of Best Models Separately and as an Ensemble')
        self._report.append(' ')
        self._report.append('Considering Best K = %d models' % k)
        self._report.append('Ensemble weighting: %s' % ensemble_weighting)


class ChartFReport(object):
    def __init__(self, k, ensemble_weighting, column_definitions):
        self._column_definitions = column_definitions
        self._report = Report()
        self._header(k, ensemble_weighting)
        cd = self._column_definitions.defs_for_columns(
            'validation_month',
            'mae_ensemble',
            'mae_best_next_month',
            'median_price',
            'fraction_median_price_next_month_ensemble',
            'fraction_median_price_next_month_best',
        )
        self._ct = ColumnsTable(columns=cd, verbose=True)

    def write(self, path):
        self._ct.append_legend()
        for line in self._ct.iterlines():
            self._report.append(line)
        self._report.write(path)

    def detail_line(self, **kwds):
        with_spaces = {
            k: (None if self._column_definitions.replace_by_spaces(k, v) else v)
            for k, v in kwds.iteritems()
        }
        self._ct.append_detail(**with_spaces)

    def _header(self, k, ensemble_weighting):
        self._report.append('Comparison of Errors of Ensemble and Best Model That Know the Future')
        self._report.append(' ')
        self._report.append('Considering Best K = %d models' % k)
        self._report.append('Ensemble weighting: %s' % ensemble_weighting)


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


def check_key_order(d):
    keys = d.keys()
    for index, key1_key2 in enumerate(zip(keys, keys[1:])):
        key1, key2 = key1_key2
        # print index, key1, key2
        mae1 = d[key1].mae
        mae2 = d[key2].mae
        trace_unless(mae1 <= mae2, 'should be non increasing',
                     index=index, mae1=mae1, mae2=mae2,
                     )


def make_charts_ef(k, reduction, actuals, median_price, control):
    '''Write charts e and f, return median-absolute-relative_regret object'''
    def interesting():
        return False
        return k == 5

    def trace_if_interesting():
        if interesting():
            pdb.set_trace()

    trace_if_interesting()
    ensemble_weighting = 'exp(-MAE/100000)'
    mae = {}
    for validation_month in control.validation_months:
        e = ChartEReport(k, ensemble_weighting, control.column_definitions)
        next_month = Month(validation_month).increment(1).as_str()
        validation_month_keys = list(reduction[validation_month].keys())
        cum_weighted_predictions = None
        cum_weights = 0
        mae_validation = None
        check_key_order(reduction[validation_month])
        # write lines for the k best individual models
        # accumulate info needed to build the ensemble model
        for index in xrange(k):
            validation_month_key = validation_month_keys[index]
            validation_month_value = reduction[validation_month][validation_month_key]
            next_month_value = reduction[next_month][validation_month_key]
            if mae_validation is not None:
                trace_unless(mae_validation <= validation_month_value.mae,
                             'should be non-decreasing',
                             mae_previous=mae_validation,
                             mae_next=validation_month_value.mae,
                             )
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
            )
            # need the mae of the ensemble
            # need the actuals and predictions? or is this already computed
            predictions_next = next_month_value.predictions
            if cum_weighted_predictions is None:
                cum_weighted_predictions = weight * predictions_next
            else:
                cum_weighted_predictions += weight * predictions_next
            cum_weights += weight
        # write line comparing the best individual model in the next month
        # to the ensemble model
        trace_if_interesting()
        ensemble_predictions = cum_weighted_predictions / cum_weights
        ensemble_rmse, ensemble_mae, ensemble_ci95_low, ensemble_ci95_high = errors(
            actuals[next_month],
            ensemble_predictions,
        )
        best_key = reduction[next_month].keys()[0]
        best_value = reduction[next_month][best_key]
        e.detail_line(
            validation_month=validation_month,
            mae_ensemble=ensemble_mae,
            mae_best_next_month=best_value.mae,
            model=best_key.model,
            n_months_back=best_key.n_months_back,
            n_estimators=best_key.n_estimators,
            max_features=best_key.max_features,
            max_depth=best_key.max_depth,
            learning_rate=best_key.learning_rate,
        )
        e.write(control.path_out_e % (k, validation_month))
        mae[validation_month] = Bunch(
            ensemble=ensemble_mae,
            best_next_month=best_value.mae,
        )
    # TODO: also create a graph
    pdb.set_trace()
    f = ChartFReport(k, ensemble_weighting, control.column_definitions)
    # TODO: add columns median_price, ensb_fmp, best_fmp
    # where fmp = mae as fraction of median price
    regrets = []
    relative_errors = []
    for validation_month in control.validation_months:
        next_month_value = reduction[next_month][validation_month_key]
        regret = mae[validation_month].ensemble - mae[validation_month].best_next_month
        regrets.append(regret)
        relative_error = regret / median_price[validation_month]
        relative_errors.append(relative_error)
        f.detail_line(
            validation_month=validation_month,
            mae_ensemble=mae[validation_month].ensemble,
            mae_best_next_month=mae[validation_month].best_next_month,
            median_price=median_price[next_month],
            fraction_median_price_next_month_ensemble=mae[validation_month].ensemble / median_price[next_month],
            fraction_median_price_next_month_best=mae[validation_month].best_next_month / median_price[next_month],
        )
    median_absolute_regret = np.median(np.abs(regrets))
    median_absolute_relative_regret = np.median(np.abs(relative_errors))
    f.write(control.path_out_f % k)
    return median_absolute_regret, median_absolute_relative_regret


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
    debug = True
    g = ChartGReport()
    ks = range(1, 31, 1)
    ks.extend([40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200])
    if control.test:
        ks = (1, 5)
    for k in ks:
        median_absolute_relative_regret = make_charts_ef(k, reduction, actuals, median_prices, control)
        if not debug:
            g.detail(k, median_absolute_relative_regret)
    if not debug:
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
    '''(reduction[validation_month][ModelDescription] = ModelResults, sorted by increasing MAE,
    all_actuals[validation_month])'''

    def process_records(path):
        '''(validation_month, model[ModelDescription] = ModelResult, actuals[validation_month])
        for the validation_month in the path'''
        def make_model_description(key):
            is_en = isinstance(key, ResultKeyEn)
            is_gbr = isinstance(key, ResultKeyGbr)
            is_rfr = isinstance(key, ResultKeyRfr)
            is_tree = is_gbr or is_rfr
            result = ModelDescription(
                model='en' if is_en else ('gb' if is_gbr else 'rf'),
                n_months_back=key.n_months_back,
                units_X=key.units_X if is_en else 'natural',
                units_y=key.units_y if is_en else 'natural',
                alpha=key.alpha if is_en else None,
                l1_ratio=key.l1_ratio if is_en else None,
                n_estimators=key.n_estimators if is_tree else None,
                max_features=key.max_features if is_tree else None,
                max_depth=key.max_depth if is_tree else None,
                loss=key.loss if is_gbr else None,
                learning_rate=key.learning_rate if is_gbr else None,
            )
            return result

        def make_model_result(value):
            rmse, mae, low, high = errors(value.actuals, value.predictions)
            result = ModelResults(
                rmse=rmse,
                mae=mae,
                ci95_low=low,
                ci95_high=high,
                predictions=value.predictions,
            )
            return result

        print 'reducing', path
        validation_month = extract_yyyymm(path)
        model = {}
        actuals = None
        n_records_retained = 0
        with open(path, 'rb') as f:
            while True:
                try:
                    record = pickle.load(f)  # read until EOF
                    assert isinstance(record, tuple), type(record)
                    key, value = record
                    n_records_retained += 1
                    # verify that each model_key occurs at most once in the validation month
                    model_key = make_model_description(key)
                    if model_key in model:
                        print '++++++++++++++++++++++'
                        print validation_month, model_key
                        print 'duplicate model key'
                        pdb.set_trace()
                        print '++++++++++++++++++++++'
                    model[model_key] = make_model_result(value)
                    # verify that each of the actuals in the validation month is the same
                    if actuals is None:
                        actuals = value.actuals
                    else:
                        if np.array_equal(actuals, value.actuals):
                            pass
                        else:
                            print '++++++++++++++++++++'
                            print 'actuals changed for a validation month'
                            print validation_month, actuals
                            pdb.set_trace()
                            print '++++++++++++++++++++'
                except ValueError as e:
                    if record is not None:
                        print record
                    print 'ignoring ValueError in record %d: %s' % (n_records_retained, e)
                    pdb.set_trace()
                except EOFError:
                    break
        print 'retained %d records in validation month %s' % (n_records_retained, validation_month)
        return validation_month, model, actuals

    reduction = collections.defaultdict(dict)
    all_actuals = {}
    paths = sorted(glob.glob(control.path_in_ege))
    for path in paths:
        validation_month, model, actuals = process_records(path)
        # sort models by increasing MAE
        sorted_models = collections.OrderedDict(sorted(model.items(), key=lambda t: t[1].mae))
        check_key_order(sorted_models)
        reduction[validation_month] = sorted_models
        all_actuals[validation_month] = actuals
        if control.debug:
            break

    return reduction, all_actuals


class Sampler(object):
    def __init__(self, fraction):
        self._fraction = fraction
        self._keys = None

    def sample(self, d):
        'return ordered dictionary using the same keys for each input dictionary d'
        if self._keys is None:
            k = int(len(d) * self._fraction)
            self._keys = random.sample(list(d), k)
        result = {}
        for key in self._keys:
            result[key] = d[key]
        sorted_result = collections.OrderedDict(sorted(result.items(), key=lambda t: t[1].mae))
        return sorted_result


def make_samples(reduction, fraction):
    'return a random sample of the reduction stratified by validation_month as an ordereddict'
    # use same keys (models) every validation month
    sampler = Sampler(fraction)
    result = {}
    for validation_month, validation_dict in reduction.iteritems():
        samples = sampler.sample(validation_dict)
        check_key_order(samples)
        result[validation_month] = sampler.sample(validation_dict)
    return result


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
        for validation_month in samples.keys():
            check_key_order(reduction[validation_month])
            check_key_order(samples[validation_month])
        lap('check key order')
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

        # check that the reduction dictionaries are ordered by mae
        for validation_month in reduction.iterkeys():
            d = reduction[validation_month]
            check_key_order(d)

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
