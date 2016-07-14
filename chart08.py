'''Compare best model MAEs by time period by feature set

INVOCATION
 python chart08.py [--data] [--test]
where
 --data cuases WORKING/chart08/data.pickle to be created
 --test causes unusable output

INPUT FILES
 WORKING/chart07/*/data.pickle  Contains best model descriptions and MAEs
 WORKING/chart08/data.pickle    Reduction of above files

OUTPUT FILES
 WORKING/chart08/a.txt         The comparison
 WORKING/chart08/data.pickle   Reduction of key input files
'''

from __future__ import division

import argparse
import collections
import cPickle as pickle
import os
import pdb
from pprint import pprint as pp
import random
import sys

from Bunch import Bunch
from chart07 import ReductionKey, ReductionValue
from ColumnsTable import ColumnsTable
from Features import Features
from Path import Path
from Report import Report
from Timer import Timer
from valavm import ResultKeyEn, ResultKeyGbr, ResultKeyRfr

if False:
    # avoid pyflakes errors
    ReductionKey
    ReductionValue
    ColumnsTable
    Features
    Report


def make_control(argv):
    'return a Bunch'
    parser = argparse.ArgumentParser()
    parser.add_argument('invocation')
    parser.add_argument('--data', action='store_true')
    parser.add_argument('--test', action='store_true')
    arg = parser.parse_args(argv)
    arg.base_name = 'chart08'

    random_seed = 123
    random.seed(random_seed)

    dir_working = Path().dir_working()
    dir_out = '%s%s/' % (dir_working, arg.base_name)
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    # fit models for these months
    test_months = (
        '200512',
        '200601', '200602', '200603', '200604', '200605', '200606',
        '200607', '200608', '200609', '200610', '200611', '200612',
        '200701', '200702', '200703', '200704', '200705', '200706',
        '200707', '200708', '200709', '200710', '200711', '200712',
        '200801', '200802', '200803', '200804', '200805', '200806',
        '200807', '200808', '200809', '200810', '200811', '200812',
        '200901', '200902',
    )
    reduced_file_name = '0data.pickle'
    reduced_file_path = dir_out + reduced_file_name

    return Bunch(
        arg=arg,
        debug=False,
        exceptions=['incomplete list of feature groups'],
        feature_groups=['s', 'sw', 'swp', 'swpn'],
        path_in_data=reduced_file_path,
        path_in_chart07_dir=dir_working + 'chart07/',
        path_out_data=reduced_file_path,
        path_out_chart_a=dir_out + 'a.txt',
        test_months=test_months,
        timer=Timer(),
    )


Value = collections.namedtuple('Value', 'mae model')


def make_data(control):
    '''return reduction[test_month][feature_group] = (mae, model)'''
    result = collections.defaultdict(dict)
    for feature_group in control.feature_groups:
        path = '%s%s-all/0data.pickle' % (control.path_in_chart07_dir, feature_group)
        counter = collections.Counter()
        input_record_number = 0
        print 'reducting data found in', path
        with open(path, 'rb') as f:
            input_record_number += 1
            while True:
                counter['attempted to read'] += 1
                input_record_number += 1
                try:
                    input_record_number += 1
                    print 'loading', path, 'record', 1
                    record = pickle.load(f)
                    d, chart07_control = record
                    for chart_07_reduction_key, chart_07_reduction_value in d.iteritems():
                        assert isinstance(chart_07_reduction_key, ReductionKey)
                        assert isinstance(chart_07_reduction_value, ReductionValue)
                        # pull out the field we want
                        test_month = chart_07_reduction_key.test_month
                        model_description = chart_07_reduction_value.model
                        mae = chart_07_reduction_value.mae
                        model = (
                            'gb' if isinstance(model_description, ResultKeyGbr) else
                            'rf' if isinstance(model_description, ResultKeyRfr) else
                            'en' if isinstance(model_description, ResultKeyEn) else
                            None)
                        print ' ', test_month, feature_group, mae, model
                        result[test_month][feature_group] = Value(
                            mae=mae,
                            model=model,
                            )
                except ValueError as e:
                    counter['ValueError'] += 1
                    print e
                    print 'ignoring ValueError for record %d' % input_record_number
                except EOFError:
                    counter['EOFError'] += 1
                    print 'stopping read at EOFError for record %d' % input_record_number
                    break
                except pickle.UnpicklingError as e:
                    counter['UnpicklingError'] += 1
                    print e
                    print 'ignoring UnpicklingError for record %d' % input_record_number
    return result


def make_chart_a(control, data):
    'return a Report'
    def make_header(report):
        report.append('Median Absolute Errors for Most Accurate Models')
        report.append('By Month')
        report.append('By Feature Group')
        report.append(' ')

    def make_details(data, control):
        'return a ColumnsTable'
        def append_feature_group_description(ct):
            ct.append_line(' ')
            ct.append_line('Features groups;')
            ct.append_line('s    : only size features')
            ct.append_line('sw   : only size and wealth features')
            ct.append_line('swp  : only size, wealth, and property features')
            ct.append_line('swpn : all features: size, wealth, property, and neighborhood')

        ct = ColumnsTable((
            ('month', 6, '%6s', ('', 'month'), 'training month'),
            ('features', 8, '%8s', ('features', 'group'), 'group of features'),
            ('model', 5, '%5s', ('best', 'model'), 'family of best model'),
            ('mae', 6, '%6.0f', ('', 'mae'), 'mae of best model in month using features'),
            ),
            verbose=True,
            )
        for month in control.test_months:
            for features in control.feature_groups:
                mae_model = data[month][features]
                ct.append_detail(
                    month=month,
                    features=features,
                    model=mae_model.model,
                    mae=mae_model.mae,
                    )
            ct.append_detail()  # blank line separates each month
        ct.append_legend()
        append_feature_group_description(ct)

        return ct

    report = Report()
    make_header(report)
    for line in make_details(data, control).iterlines():
        report.append(line)
    return report


def make_charts(control, data):
    return {
        'a': make_chart_a(control, data),
        }


def main(argv):
    control = make_control(argv)
    print control

    # do the work
    if control.arg.data:
        data = make_data(control)
        control.timer.lap('make data reduction')
        with open(control.path_out_data, 'wb') as f:
            pickle.dump((data, control), f)
            control.timer.lap('write reduction')
    else:
        with open(control.path_in_data, 'rb') as f:
            pickled = pickle.load(f)
            data, reduction_control = pickled
        charts = make_charts(control, data)
        control.timer.lap('make charts')
        # write the charts
        for chart_key, chart_value in charts.iteritems():
            if chart_key == 'a':
                chart_value.write(control.path_out_chart_a)
            else:
                print 'bad chart_key', chart_key
                pdb.set_trace()
        control.timer.lap('write charts')

    # wrap up
    print control
    if control.arg.test:
        print 'DISCARD OUTPUT: test'
    if control.debug:
        print 'DISCARD OUTPUT: debug'
    print 'done'


if __name__ == '__main__':
    if False:
        # avoid pyflakes warnings
        pdb.set_trace()
        pp()

    main(sys.argv)
