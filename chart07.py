'''Determine most important features for the very best K models in each test month

valavm.py didn't save the fitted models, because that would have created a lot
of data.  So this program re-fits the model, in order to gain access to the
scikit-learn feature_importances_ attribute.

INVOCATION
 python chart07.py FEATURESGROUP-HPS [--data] [-test]
where
 FEATUREGROUPS is one of {s, sw, swp, swpn}
 HPS is one of {all, best1}
 FH  is FEATURESGROUP-HPS
 --data  causes WORKING/chart06/FH/data.pickle to be created
 --test  causes non-production behavior

INPUTS FILE
 WORKING/chart06/FH/data.pickle  Defines results of models for FH
 WORKING/chart07/FH/data.pickle  the reduction

OUTPUTS FILES
 WORKING/chart07/FH/data.pickle
 WORKING/chart06/FH/a-nbest-POSTIVEINT-nworst-POSITIVEINT.txt
 WORKING/chart06/FH/b.txt
'''

from __future__ import division

import argparse
import collections
import cPickle as pickle
import numpy as np
import os
import pandas as pd
import pdb
from pprint import pprint as pp
import random
import sys

import arg_type
from AVM import AVM
from Bunch import Bunch
from chart06 import ModelDescription, ModelResults, ColumnDefinitions, errors
from ColumnsTable import ColumnsTable
from Features import Features
from Path import Path
from Report import Report
from Timer import Timer
from valavm import ResultKeyEn, ResultKeyGbr, ResultKeyRfr, ResultValue
# cc = columns_contain

# use valavm imports so as to avoid an error message from pyflakes
if False:
    print ResultKeyEn
    print ResultKeyGbr
    print ResultKeyRfr
    print ResultValue


def make_control(argv):
    'return a Bunch'
    parser = argparse.ArgumentParser()
    parser.add_argument('invocation')
    parser.add_argument('features_hps', type=arg_type.features_hps)
    parser.add_argument('--data', action='store_true')
    parser.add_argument('--test', action='store_true')
    arg = parser.parse_args(argv)
    arg.base_name = 'chart07'

    random_seed = 123
    random.seed(random_seed)

    dir_working = Path().dir_working()
    dir_out = dir_working + arg.base_name + '/' + arg.features_hps + '/'
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
    reduced_file_name = 'data.pickle'

    return Bunch(
        arg=arg,
        debug=False,
        k=1,  # number of best models examined
        path_in_data=dir_out + reduced_file_name,
        path_in_valavm_dir=dir_working + ('valavm/%s/' % arg.features_hps),
        path_out_data=dir_out + reduced_file_name,
        path_out_chart_a_template=dir_out + 'a-nbest-%d-nworst-%d.txt',
        path_out_chart_b=dir_out + 'b.txt',
        test_months=test_months,
        timer=Timer(),
    )


# the reduction is a dictionary
ReductionKey = collections.namedtuple('ReductionKey', 'test_month rank_index')
ReductionValue = collections.namedtuple('ReductionValue', 'model importances')


def make_chart_b(control, data):
    'return a Report'
    def make_header(report):
        report.append('Mean Probability of a Feature Being Included in a Decision Tree')
        report.append('Across the Entire Ensemble of Decisions Trees')
        report.append('For Most Accurate Model in Each Training Month')
        report.append(' ')

    def make_mean_importance_by_feature(test_months):
        'return dict[feature_name] = float, the mean importance of the feature'
        feature_names = Features().ege_names()
        mean_importance = {}  # key = feature_name
        for feature_index, feature_name in enumerate(feature_names):
            # build vector of feature_importances for feature_name
            feature_importances = np.zeros(len(test_months))
            for month_index, test_month in enumerate(test_months):
                month_importances = data[test_month]  # for each feature
                feature_importances[month_index] = month_importances[feature_index]
            mean_importance[feature_name] = np.mean(feature_importances)
        return mean_importance

    def make_details(data, test_months):
        'return a ColumnTable'
        columns_table = ColumnsTable((
            ('mean_prob', 5, '%5.2f', ('mean', 'prob'), 'mean probability feature appears in a decision tree'),
            ('feature_name', 40, '%40s', (' ', 'feature name'), 'name of feature'),
            ),
            verbose=True)
        mean_importance = make_mean_importance_by_feature(test_months)
        for feature_name in sorted(mean_importance, key=mean_importance.get, reverse=True):
            columns_table.append_detail(
                mean_prob=mean_importance[feature_name] * 100.0,
                feature_name=feature_name,
            )
        columns_table.append_legend()
        return columns_table

    report = Report()
    make_header(report)
    details = make_details(data, control.test_months)
    for line in details.iterlines():
        report.append(line)
    return report


def make_chart_a(control, data):
    'return dict[(n_best, n_worst]) --> a Report'
    def make_header(report):
        report.append('Mean Probability of a Feature Being Included in a Decision Tree')
        report.append('Across the Entire Ensemble of Decisions Trees')
        report.append('For Most Accurate Model in Each Training Month')
        report.append(' ')

    def make_details(data, test_months, n_best, n_worst):
        'return a ColumnTable'
        feature_names = Features().ege_names()
        columns_table = ColumnsTable((
            ('test_month', 6, '%6s', ('test', 'month'), 'test month'),
            ('nth', 2, '%2d', (' ', 'n'), 'rank of feature (1 ==> more frequently included)'),
            ('probability', 4, '%4.1f', (' ', 'prob'), 'probability feature appears in a decision tree'),
            ('feature_name', 40, '%40s', (' ', 'feature name'), 'name of feature'),
            ),
            verbose=True)
        for test_month in test_months:
            importances = data[test_month]
            sorted_indices = importances.argsort()  # sorted first lowest, last highest
            for nth_best in xrange(n_best):
                index = sorted_indices[len(importances) - nth_best - 1]
                columns_table.append_detail(
                    test_month=test_month,
                    nth=nth_best + 1,
                    probability=importances[index] * 100.0,
                    feature_name=feature_names[index]
                    )
            for nth in xrange(n_worst):
                nth_worst = n_worst - nth - 1
                index = sorted_indices[nth_worst]
                columns_table.append_detail(
                    test_month=test_month,
                    nth=len(importances) - nth_worst,
                    probability=importances[index] * 100.0,
                    feature_name=feature_names[index]
                    )
            if n_best > 1 or n_worst > 1:
                # insert blank line between test_months if more than 1 row in a month
                columns_table.append_detail()
        columns_table.append_legend()
        return columns_table

    def make_report(n_best, n_worst):
        report = Report()
        make_header(report)
        details = make_details(data, control.test_months, n_best, n_worst)
        for line in details.iterlines():
            report.append(line)
        return report

    reports = {}

    def add_report(n_best, n_worst):
        reports[(n_best, n_worst)] = make_report(n_best, n_worst)

    add_report(1, 0)
    add_report(10, 0)
    add_report(15, 0)
    add_report(0, 10)
    add_report(0, 20)
    add_report(0, 40)
    add_report(0, 60)

    return reports


def make_charts(control, data):
    'return dict of charts'
    # all models are fit to an X matrix with the same features in the same columns
    if control.debug:
        return {'chart_b': make_chart_b(control, data)}
    assert control.k == 1, control  # this code works only for the very best model
    chart_a = make_chart_a(control, data)
    chart_b = make_chart_b(control, data)
    return {
        'chart_a': chart_a,
        'chart_b': chart_b,
        }


def make_data(control):
    'return dict[test_month] = coefficients_or_feature_importances'
    result = {}
    for test_month in control.test_months:
        path = '%s%s-%s.pickle' % (
            control.path_in_valavm_dir,
            control.arg.features_hps,
            test_month,
            )
        print 'make_data reading', path
        assert control.k == 1
        with open(path, 'rb') as f:
            # read each fitted model and keep the k best
            lowest_mae = None
            best_key = None
            best_importances = None
            counter = collections.Counter()
            input_record_number = 0
            while True:
                counter['attempted to read'] += 1
                input_record_number += 1
                try:
                    record = pickle.load(f)
                    key, value = record
                    actuals_predictions, importances = value
                    actuals = actuals_predictions.actuals
                    predictions = actuals_predictions.predictions
                    rmse, mae, ci95_low, ci95_high = errors(actuals, predictions)
                    if (lowest_mae is None) or (mae < lowest_mae):
                        lowest_mae = mae
                        best_key = key
                        best_importances = importances
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
            print 'test_month', test_month, 'type(best_key)', type(best_key)
            print
            key = ReductionKey(test_month, 0)
            value = ReductionValue(best_key, best_importances)
            result[key] = value
    return result


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
            if chart_key == 'chart_a':
                # white chart_a's
                for chart_a_key, chart_a_value in chart_value.iteritems():
                    n_best, n_worst = chart_a_key
                    path = control.path_out_chart_a_template % (n_best, n_worst)
                    print 'writing', path
                    chart_a_value.write(path)
            elif chart_key == 'chart_b':
                path = control.path_out_chart_b
                print 'writing', path
                chart_value.write(path)
            else:
                print 'bad chart key', chart_key
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
        pd.DataFrame()
        ModelDescription
        ModelResults
        ColumnDefinitions
        AVM()

    main(sys.argv)
