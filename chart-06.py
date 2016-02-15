'''create charts showing results of valgbr.py

INVOCATION
  python chart-06.py --valavm VALAVM [--data] [--test]

INPUT FILES
 WORKING/chart-01/data.pickle
 WORKING/samples-train.csv
 WORKING/valavm-IN_DIRECTORY/YYYYMM.pickle

OUTPUT FILES
 WORKING/chart-06-VALAVM/[test-]data.pickle    | reduced data
 WORKING/chart-06-VALAVM/[test-]a.pdf          | range of losses by model (graph)
 WORKING/chart-06-VALAVM/[test-]b-YYYYMM.txt   | HPs with lowest losses
 WORKING/chart-06-VALAVM/[test-]c.pdf          | best model each month
 WORKING/chart-06-VALAVM/[test-]d.pdf          | best & 50th best each month
 WORKING/chart-06-VALAVM/[test-]e.pdf          | best 50 models each month (was chart-07)
 WORKING/chart-06-VALAVM/[test-]best.pickle    | dataframe with best choices each month
 WORKING/chart-06-VALAVM/[test-]log.txt        | log file (created by print statements)
'''

from __future__ import division

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

ModelResult = collections.namedtuple('ModelResult', 'mae')


def usage(msg=None):
    print __doc__
    if msg is not None:
        print msg
    sys.exit(1)


def make_control(argv):
    # return a Bunch

    print argv
    if len(argv) not in (2, 3, 4):
        usage('invalid number of arguments')

    pcl = ParseCommandLine(argv)
    arg = Bunch(
        base_name='chart-06',
        data=pcl.has_arg('--data'),
        test=pcl.has_arg('--test'),
        valavm=pcl.get_arg('--valavm'),
    )

    random_seed = 123
    random.seed(random_seed)

    dir_working = Path().dir_working()

    debug = False

    test = 'test-' if arg.test else ''

    # assure output directory exists
    dir_out = dir_working + arg.base_name + '-' + arg.valavm + '/'
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    return Bunch(
        arg=arg,
        debug=debug,
        path_in_ege=dir_working + 'valavm-' + arg.valavm + '/??????.pickle',
        path_in_samples=dir_working + 'samples-train.csv',
        path_out_a=dir_out + test + 'a.pdf',
        path_out_b=dir_out + test + 'b-%d.txt',
        path_out_cd=dir_out + test + '%s.txt',
        path_out_d=dir_out + test + 'd.txt',
        path_out_e=dir_out + test + 'e.txt',
        path_data=dir_out + test + 'data.pickle',
        path_out_best=dir_out + test + 'best.pickle',
        path_out_log=dir_out + 'log.txt',
        path_in_median_prices=dir_working + 'chart-01/data.pickle',
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


def make_chart_b(reduction, control):
    '''MAE for best-performing models by training period
    NOTE: This code knows that en is never the best model
    '''
    def report(year, month):
        format_header = '%6s %5s %2s %4s %4s %4s %-20s'
        format_detail = '%6d %5s %2d %4d %4d %4s %-20s'
        format_detail_no_max_depth = '%6d %5s %2d %4s %4d %4s %-20s'

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


class ChartReport(object):
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
    cr = ChartReport()
    for test_month in test_months():
        median_price = median_prices[test_month]
        sorted_hps_test_month = sorted_hps[test_month]
        for dl in detail_lines(sorted_hps_test_month):
            cr.detail_line(dl, test_month, sorted_hps_test_month, median_price)
    cr.write(control.path_out_cd % report_id)
    return


def make_median_prices(medians):
    return {test_month: medians[test_month.as_int()]
            for test_month in test_months()
            }


class ChartEReport(object):
    def __init__(self, k, ensemble_weighting):
        self.report = Report()
        self.format_header = '%6s %20s %6s %6s %6s'
        self.format_model = '%6d %20s %6.0f %6.4f %6.0f'
        self.format_ensemble = '%6s %20s %6s %6s %6.0f'
        self._header(k, ensemble_weighting)

    def _header(self, k, ensemble_weighting):
        self.report.append('Performance of Best Models Separately and as an Ensemble')
        self.report.append(' ')
        self.report.append('Consider best %d models' % k)
        self.report.append('Ensemble weighting: %s' % ensemble_weighting)
        self.report.append(' ')
        self.report.append(self.format_header % ('test', 'best', 'test', '', 'next'))
        self.report.append(self.format_header % ('month', 'model', 'MAE', 'weight', 'MAE'))

    def model_detail(self, month, model_s, test_mae, weight, next_mae):
        self.report.append(self.format_model % (month, model_s, test_mae, weight, next_mae))

    def ensemble_detail(self, ensemble_mae):
        self.report.append(self.format_ensemble % (' ', 'ensemble', 'na', 'na', ensemble_mae))


def model_to_str(k):
    if isinstance(k, ResultKeyEn):
        return 'en-%d-%s-%s-%d-%d' % (
            k.n_months_back, k.units_X, k.units_y, k.alpha, k.l1_ratio)
    elif isinstance(k, ResultKeyGbr):
        assert k.loss == 'ls', k
        if k.max_depth is None:
            return 'gb-%0d-%d-%s-%.1f' % (
                k.n_months_back, k.n_estimators, 'None', k.learning_rate)
        else:
            return 'gb-%0d-%d-%d-%.1f' % (
                k.n_months_back, k.n_estimators, k.max_depth, k.learning_rate)
    elif isinstance(k, ResultKeyRfr):
        if k.max_depth is None:
            return 'rf-%0d-%d-%s' % (
                k.n_months_back, k.n_estimators, 'None')
        else:
            return 'rf-%0d-%d-%d' % (
                k.n_months_back, k.n_estimators, k.max_depth)
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


def make_chart_e(actuals_d, predictions_d, mae_d, control, sorted_hpsOLD):
    if len(actuals_d) != len(predictions_d) != len(mae_d):
        pdb.trace()
    k = 10  # use k best models
    ensemble_weighting = 'exp(-MAE/100000)'
    r = ChartEReport(k, ensemble_weighting)
    for test_month in test_months():
        if test_month == 200711:
            # we didn't test a model on 200712
            break
        test_month_mae = mae_d[test_month]
        models_sorted = sorted(test_month_mae, key=test_month_mae.get)  # keys in order of mae
        weights = []
        actuals = []
        predictions = []
        for index in xrange(k):
            model = models_sorted[index]
            next_month = Month(test_month).increment(1).as_int()
            test_mae = mae_d[test_month][model]
            next_mae = mae_d[next_month][model]
            weight = math.exp(-test_mae / 100000.0)
            r.model_detail(test_month, model_to_str(model), test_mae, weight, next_mae)
            weights.append(weight)
            actuals.append(actuals_d[test_month][model])
            predictions.append(predictions_d[test_month][model])
            if control.test:
                break
        check_actuals(actuals)
        ensemble_actuals = actuals[0]  # they are all the same, so pick one
        ensemble_predictions = make_ensemble_predictions(predictions, weights)
        ensemble_rmse, ensemble_mae = errors(ensemble_actuals, ensemble_predictions)
        r.ensemble_detail(ensemble_mae)


def make_charts(reduction_df, actuals_d, predictions_d, mae_d, control, median_prices):
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

    def detail_lines_d(sorted_hps_test_month):
        result = []
        for k in xrange(len(sorted_hps_test_month)):
            result.append(k)
            if k == 4:
                break
        result.append(len(sorted_hps_test_month) - 1)
        return result

    make_chart_cd(reduction_df, control, median_prices, sorted_hps, detail_lines_c, 'c')
    make_chart_cd(reduction_df, control, median_prices, sorted_hps, detail_lines_d, 'd')
    make_chart_e(actuals_d, predictions_d, mae_d, control, sorted_hps)


def errors(actuals, predictions):
    'return root_mean_squared_error, median_absolute_error'
    errors = actuals - predictions
    root_mean_squared_error = np.sqrt(np.sum(errors * errors) / (1.0 * len(errors)))
    median_absolute_error = np.median(np.abs(errors))
    return root_mean_squared_error, median_absolute_error


def extract_yyyymm(path):
    return path.split('/')[4].split('.')[0]


def make_data(control):
    'return reduction data frame, reduction dict, ege_control'

    actuals_d = collections.defaultdict(dict)
    predictions_d = collections.defaultdict(dict)
    mae_d = collections.defaultdict(dict)

    def make_row(test_month, k, v):
        'return nex row of dataframe; add to reduction dictionaries'
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
    count = 0
    for path in paths:
        process_records(path, rows_list)
        count += 1
        if control.test and count == 2:
            break
    return pd.DataFrame(rows_list), actuals_d, predictions_d, mae_d


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(logfile_path=control.path_out_log)
    print control

    if control.arg.data:
        df, actuals_d, predictions_d, mae_d = make_data(control)
        with open(control.path_data, 'wb') as f:
            pickle.dump((df, actuals_d, predictions_d, mae_d, control), f)
        print 'wrote reduction data file'
    else:
        with open(control.path_data, 'rb') as f:
            reduction_df, actuals_d, predictions_d, mae_d, reduction_control = pickle.load(f)
            with open(control.path_in_median_prices, 'rb') as g:
                data, reduction_control = pickle.load(g)
                counts, means, medians, prices = data
                make_charts(reduction_df, actuals_d, predictions_d, mae_d, control, medians)

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
