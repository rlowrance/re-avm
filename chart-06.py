'''create charts showing results of valgbr.py

INVOCATION
  python chart-06.py [--data] [--test]

INPUT FILES
 INPUT/valavm/YYYYMM.pickle

OUTPUT FILES
 WORKING/chart-06/[test-]data.pickle
 WORKING/chart-06/[test-]2007-a.pdf    comparison of losses by model by month in 2007
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
        path_in_ege=dir_working + 'valavm/*.pickle',
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


def make_chart_a(df, control, ege_control):
    'write one txt file for each n_months_back'
    print 'make_chart_a'
    year = 2007

    def make_subplot(month):
        'mutate the default axes'
        print 'make_subplot month', month
        for model in set(df.model):
            subset_sorted = select_and_sort(df, year, month, model)
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


def make_chart_b(df, control, ege_control, chart_letter):
    '''write report: mae, model, HPs for year, month
    NOTE: This code knows that en is never the best model
    '''
    def report(year, month):
        format_header = '%6s %5s %2s %3s %4s %4s %-20s'
        format_detail = '%6d %5s %2d %3d %4d %4s %-20s'
        n_detail_lines = 50

        def header(r):
            r.append('MAE for %d best-performing models and their hyperparameters' % n_detail_lines)
            r.append('Year %d Month %0d' % (year, month))
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
            df.yyyymm == yyyymm
        )
        subset = df.loc[mask]
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

    year = 2007
    month_seq = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    for month in month_seq:
        report(year, month)


def make_chart_c(df, control, ege_control):
    '''write report: mae, model, HPs for month in year 2007
    NOTE: This code knows that en is never the best model
    '''
    def report(year):
        format_header = '%3s %6s %5s %2s %3s %4s %4s %-20s'
        format_detail = '%3s %6d %5s %2d %3d %4d %4s %-20s'
        month_names = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')

        def header(r):
            r.append('MAE by month for best-performing models and their hyperparameters')
            r.append('Year %0d' % year)
            r.append(' ')
            r.append(format_header % ('Mnt', 'MAE', 'Model', 'bk', 'mxd', 'nest', 'mxft', 'Other HPs'))

        def write_details(r, month_number, series):
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
                month_names[month_number],
                series.mae, series.model, series.n_months_back,
                series.max_depth,
                series.n_estimators,
                series.max_features,
                hp_string))

        def footer(r):
            r.append(' ')
            r.append('column legend:')
            r.append('mnt -> month')
            r.append('bk -> number of months back for training')
            r.append('mxd -> max depth of individual tree')
            r.append('nest -> n_estimators (number of trees)')
            r.append('mxft -> max number of features considered when selecting new split variable')

        def append_month_line(r, month):
            yyyymm = str(year * 100 + (month + 1))
            mask = (
                df.yyyymm == yyyymm
            )
            subset = df.loc[mask]
            assert len(subset) > 0, subset.shape
            subset_sorted = subset.sort_values('mae')
            write_details(r, month, subset_sorted.iloc[0])  # print value with lowest error

        r = Report()
        header(r)
        for month_number in xrange(12):
            append_month_line(r, month_number)
        footer(r)
        r.write(control.path_chart_base + str(year) + '-c.txt')

    year = 2007
    report(year)


def make_charts(df, control, ege_control):
    # make_chart_a(df, control, ege_control)
    # make_chart_b(df, control, ege_control)
    make_chart_c(df, control, ege_control)


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
            model = 'en' if is_en else ('gb' if is_gbr else 'rf')
            actuals = v.actuals.values
            predictions = v.predictions
            if predictions is None:
                print k
                print 'predictions is missing'
                pdb.set_trace()
            rmse, mae = errors(actuals, predictions)
            row = {
                'model': model,
                'yyyymm': yyyymm,
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
        ResultValue

    main(sys.argv)
