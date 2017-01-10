'''create charts showing results of valgbr.py

INVOCATION
  python chart-07.py [--data] [--test]

INPUT FILES
 WORKING/chart-06/best.pickle
 WORKING/samples-train.csv

OUTPUT FILES
 WORKING/chart-07/[test-]data.pickle
 WORKING/chart-07/[test-]2007.pdf
'''

from __future__ import division

import collections
import cPickle as pickle
import matplotlib.pyplot as plt
import numbers
import numpy as np
import os
import pandas as pd
import pdb
from pprint import pprint
import random
import sys

from AVM import AVM
from Bunch import Bunch
from chart_01_datakey import DataKey
from columns_contain import columns_contain
import layout_transactions
from Logger import Logger
from ParseCommandLine import ParseCommandLine
from Path import Path
from Report import Report
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
        base_name='chart-07',
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
        path_in_best=dir_working + 'chart-06/best.pickle',
        path_in_samples='../data/working/samples-train.csv',
        path_reduction=dir_path + reduced_file_name,
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
    year = 2007

    def make_subplot(month):
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
        row_seq = (1, 2, 3, 4)
        col_seq = (1, 2, 3)
        for row in row_seq:
            for col in col_seq:
                axes_number += 1  # count across rows (axes_number == month number)
                plt.subplot(len(row_seq), len(col_seq), axes_number)
                make_subplot(axes_number)
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


def make_charts_c(reduction, control, median_prices):
    '''write report: mae, model, HPs for month in year 2007
    NOTE: This code knows that en is never the best model
    '''
    def create_report():
        format_header = '%8s %6s %8s %5s %2s %3s %4s %4s %-20s'
        format_detail = ' %4d-%02d %6d %8d %5s %2d %3d %4d %4s %-20s'

        def header(r):
            r.append('MAE by month for best-performing models and their hyperparameters')
            #  r.append('Year %0d' % year)
            r.append(' ')
            r.append(format_header % ('Training', '', '', '', '', '', '', '', ''))
            r.append(format_header % ('Period', 'MAE', 'MedPrice', 'Model', 'bk', 'mxd', 'nest', 'mxft', 'Other HPs'))

        def write_details(r, year, month, series):
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
            median_price = median_prices[DataKey(year, month)]
            r.append(format_detail % (
                year, month,
                series.mae, median_price,
                series.model, series.n_months_back,
                series.max_depth,
                series.n_estimators,
                series.max_features,
                hp_string))
            return median_price

        def footer(r):
            r.append(' ')
            r.append('column legend:')
            r.append('Period -> year-month')
            r.append('MAE -> median absolute error in the price estimate')
            r.append('MedPrice -> median price in the month')
            r.append('bk -> number of months back for training')
            r.append('mxd -> max depth of individual tree')
            r.append('nest -> n_estimators (number of trees)')
            r.append('mxft -> max number of features considered when selecting new split variable')

        def append_month_line(r, year, month):
            yyyymm = str(year * 100 + month)
            mask = (
                reduction.yyyymm == yyyymm
            )
            subset = reduction.loc[mask]
            if len(subset) == 0:
                print 'no data for', yyyymm
                return None, None
            subset_sorted = subset.sort_values('mae')
            best_series = subset_sorted.iloc[0]  # the one with the lowest MAE error
            median_price = write_details(r, year, month, best_series)
            return best_series, median_price

        r = Report()
        header(r)
        best = {}
        for year in (2006, 2007):
            months = (
                (12,) if year == 2006 else
                (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12) if year == 2007 else
                (1,)
            )
            for month in months:
                best_series, median_price = append_month_line(r, year, month)
                if best_series is not None:
                    best[best_series.yyyymm] = (best_series, median_price)
        footer(r)
        r.write(control.path_chart_base + str(year) + '-c.txt')
        with open(control.path_chart_base + 'best.pickle', 'wb') as f:
            pickle.dump(best, f)
        return best

    return create_report()


def make_charts_d(samples, control, median_prices, best):
    '''write report: mae, model, HPs for month in year 2007
    NOTE: This code knows that en is never the best model
    '''
    def create_report():
        format_header = '%7s %5s %5s %6s %5s %5s'
        format_detail = '%4d-%02d %5d %5d %6d %5s %05d'

        def header(r):
            r.append('Test and Train MAE by month for best performing models')
            r.append(' ')
            r.append(format_header % ('test', 'test', 'train', 'median', 'best', 'model'))
            r.append(format_header % ('period', 'MAE', 'MAE', 'price', 'model', 'bk'))

        def footer(r):
            r.append(' ')
            r.append('column legend:')
            r.append('Period -> year-month')
            r.append('MAE -> median absolute error in the price estimate')
            r.append('bk -> number of months back for training')

        def test_mae(year, month, best_series):
            'return MAE from training and testing specified model for period year-month'
            def time_period(year, month):
                return year * 100 + month

            def isnan(x):
                return x != x

            def error(msg):
                assert False, msg

            def median_absolute_error(actuals, predictions):
                errors = actuals - predictions
                abs_errors = errors.abs()
                median_abs_error = abs_errors.median()
                return median_abs_error

            def toint(x):
                'convert to int if a number'
                return int(x) if isinstance(x, numbers.Number) else x

            assert best_series.model == 'gb', best_series
            print year, month
            print best_series
            model = AVM(
                model_name='GradientBoostingRegressor',
                forecast_time_period=time_period(year, month),
                n_months_back=best_series.n_months_back,
                random_state=control.random_seed,
                alpha=None if isnan(best_series.alpha) else error('alpha'),
                l1_ratio=None if isnan(best_series.l1_ratio) else error('l1_ratio'),
                units_X=best_series.units_X,
                units_y=best_series.units_y,
                n_estimators=toint(best_series.n_estimators),
                max_depth=toint(best_series.max_depth),
                max_features=toint(best_series.max_features),
                learning_rate=best_series.learning_rate,
                loss=best_series.loss,
            )
            # mimic valavm::fit_and_run
            # convert the transaction month column to int
            # that's required by the fit method
            samples[layout_transactions.yyyymm] = samples[layout_transactions.yyyymm].astype(int)

            # fit isn't determining the correct training set
            model.fit(samples)
            mask = samples[layout_transactions.yyyymm] == time_period(year, month)
            samples_test = samples[mask]
            predictions = model.predict(samples_test)
            if predictions is None:
                print 'no predictions'
                pdb.set_trace()
            actuals = samples_test[layout_transactions.price]
            return median_absolute_error(actuals, predictions)

        def append_month_line(r, year, month):
            best_key = str(year * 100 + month)
            best_series, median_price = best[best_key]
            r.append(format_detail % (
                year,
                month,
                best_series.mae,
                test_mae(year, month, best_series),
                median_price,
                best_series.model,
                best_series.n_months_back,
            ))

        r = Report()
        header(r)
        for year in (2007,):
            months = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
            for month in months:
                append_month_line(r, year, month)
        footer(r)
        r.write(control.path_chart_base + str(year) + '-d.txt')

    create_report()


def make_charts(reduction, samples, control, median_prices):
    # make_chart_a(reduction, control)
    # make_chart_b(reduction, control)
    best = make_charts_c(reduction, control, median_prices)
    make_charts_d(samples, control, median_prices, best)


def errors(actuals, predictions):
    'return root_mean_squared_error, median_absolute_error'
    errors = actuals - predictions
    root_mean_squared_error = np.sqrt(np.sum(errors * errors) / (1.0 * len(errors)))
    median_absolute_error = np.median(np.abs(errors))
    return root_mean_squared_error, median_absolute_error


def extract_yyyymm(path):
    return path.split('/')[4].split('.')[0]


def test_mae(year, month, best_series, samples, control):
    'return MAE from training and testing specified model for period year-month'
    def time_period(year, month):
        return year * 100 + month

    def isnan(x):
        return x != x

    def error(msg):
        assert False, msg

    def median_absolute_error(actuals, predictions):
        errors = actuals - predictions
        abs_errors = errors.abs()
        median_abs_error = abs_errors.median()
        return median_abs_error

    def toint(x):
        'convert to int if a number'
        return int(x) if isinstance(x, numbers.Number) else x

    pdb.set_trace()
    assert best_series.model == 'gb', best_series
    print year, month
    print best_series
    model = AVM(
        model_name='GradientBoostingRegressor',
        forecast_time_period=time_period(year, month),
        n_months_back=best_series.n_months_back,
        random_state=control.random_seed,
        alpha=None if isnan(best_series.alpha) else error('alpha'),
        l1_ratio=None if isnan(best_series.l1_ratio) else error('l1_ratio'),
        units_X=best_series.units_X,
        units_y=best_series.units_y,
        n_estimators=toint(best_series.n_estimators),
        max_depth=toint(best_series.max_depth),
        max_features=toint(best_series.max_features),
        learning_rate=best_series.learning_rate,
        loss=best_series.loss,
    )
    # mimic valavm::fit_and_run
    # convert the transaction month column to int
    # that's required by the fit method
    samples[layout_transactions.yyyymm] = samples[layout_transactions.yyyymm].astype(int)

    # fit isn't determining the correct training set
    model.fit(samples)
    mask = samples[layout_transactions.yyyymm] == time_period(year, month)
    samples_test = samples[mask]
    predictions = model.predict(samples_test)
    if predictions is None:
        print 'no predictions'
        pdb.set_trace()
        actuals = samples_test[layout_transactions.price]
        return median_absolute_error(actuals, predictions)


def make_data(control, best, samples):
    'return MAEs for testing of models'
    pdb.set_trace()
    mae = {}
    for year in (2007,):
        for month in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12):
            yyyymm = str(year * 100 + month)
            best_series, unknown = best[yyyymm]
            best_model, mae = test_mae(year, month, best_series, samples, control)
            mae[(year, month)] = (best_series, mae)
    return mae


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(base_name=control.arg.base_name)
    print control

    if control.arg.data:
        with open(control.path_in_best, 'rb') as f:
            pdb.set_trace()
            best = pickle.load(f)
        samples = pd.read_csv(control.path_in_samples)
        mae = make_data(control, best, samples)
        with open(control.path_reduction, 'wb') as f:
            pickle.dump((mae, control), f)
    else:
        with open(control.path_reduction, 'rb') as f:
            mae, reduction_control = pickle.load(f)
            make_chart(mae, samples, control)

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
