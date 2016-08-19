'''Determine most important features for the very best K models in each test month
valavm.py didn't save the fitted models, because that would have created a lot
of data.  So this program re-fits the model, in order to gain access to the
scikit-learn feature_importances_ attribute.
INVOCATION
 python chart07.py {features_group}-{hps}-{locality} [--data] [-test]
where
 features_groput is one of {s, sw, swp, swpn}
 hps is one of {all, best1}
 locality is in {'census', 'city,' 'global', 'zip'}
 --data  causes WORKING/chart06/FHL/0data.pickle to be created
 --test  causes non-production behavior
INPUTS FILE
 WORKING/valavm/{features_group}-{hps}-{locality}/{validation_month}.pickle
 WORKING/chart07/{features_group}-{hps}-{locality}/0data.pickle  the reduction
OUTPUTS FILES
 WORKING/chart07/{features_group}-{hps}-{locality}/0data.pickle
 WORKING/chart07/{features_group}-{hps}-{locality}/a-nbest-POSTIVEINT-nworst-POSITIVEINT.txt
 WORKING/chart07/{features_group}-{hps}-{locality}/b.txt
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
from chart06types import ModelDescription, ModelResults, ColumnDefinitions
from chart07types import ReductionKey, ReductionValue
from ColumnsTable import ColumnsTable
import errors
from Features import Features
from Path import Path
from Report import Report
from Timer import Timer
from valavmtypes import ResultKeyEn, ResultKeyGbr, ResultKeyRfr, ResultValue
import matplotlib.pyplot as plt

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
    parser.add_argument('features_hps_locality', type=arg_type.features_hps_locality)
    parser.add_argument('--data', action='store_true')
    parser.add_argument('--test', action='store_true')
    arg = parser.parse_args(argv)
    arg.base_name = 'chart07'
    arg.features, arg.hps, arg.locality = arg.features_hps_locality.split('-')

    random_seed = 123
    random.seed(random_seed)

    dir_working = Path().dir_working()
    dir_out = dir_working + arg.base_name + '/' + arg.features_hps_locality + '/'
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

    return Bunch(
        arg=arg,
        debug=False,
        k=1,  # number of best models examined
        path_in_data=dir_out + reduced_file_name,
        path_in_valavm_dir=dir_working + ('valavm/%s/' % arg.features_hps_locality),
        path_out_data=dir_out + reduced_file_name,
        path_out_chart_a_template=dir_out + 'a-nbest-%d-nworst-%d.txt',
        path_out_chart_a_pdf=dir_out + 'a-nbest-%d-nworst-%d.pdf',
        path_out_chart_b=dir_out + 'b.txt',
        path_out_chart_b_pdf=dir_out + 'b.pdf',
        test_months=test_months,
        timer=Timer(),
    )


# the reduction is a dictionary

def make_chart_b(control, data):
    'return a Report'
    def make_header(report):
        report.append('Mean Probability of a Feature Being Included in a Decision Tree')
        report.append('Across the Entire Ensemble of Decisions Trees')
        report.append('For Most Accurate Model in Each Training Month')
        report.append(' ')

    def make_mean_importance_by_feature(test_months):
        'return dict[feature_name] = float, the mean importance of the feature'
        feature_names = Features().ege_names(control.arg.features)
        mean_importance = {}  # key = feature_name
        for feature_index, feature_name in enumerate(feature_names):
            # build vector of feature_importances for feature_name
            feature_importances = np.zeros(len(test_months))  # for feature_name
            for month_index, test_month in enumerate(test_months):
                month_importances = data[ReductionKey(test_month)]  # for each feature
                all_feature_importances = month_importances.importances['feature_importances']
                if 'feature_importances' not in month_importances.importances:
                    print 'chart b sees an unexpected ensemble model'
                    print 'test_month', test_month
                    print 'month_importances', month_importances
                    print 'entering debugger'
                    pdb.set_trace()
                feature_importances[month_index] = all_feature_importances[feature_index]
            mean_importance[feature_name] = np.mean(feature_importances)
        return mean_importance

    def make_details(data, test_months):
        'return a ColumnTable'
        columns_table = ColumnsTable((
            ('mean_prob', 5, '%5.2f', ('mean', 'prob'), 'mean probability feature appears in a decision tree'),
            ('feature_name', 40, '%40s', (' ', 'feature name'), 'name of feature'),
            ),
            verbose=True)
        my_prob = []
        my_featname = []
        mean_importance = make_mean_importance_by_feature(test_months)
        for feature_name in sorted(mean_importance, key=mean_importance.get, reverse=True):
            columns_table.append_detail(
                mean_prob=mean_importance[feature_name] * 100.0,
                feature_name=feature_name,
            )
            if mean_importance[feature_name] * 100.0 >= 1:
                my_prob.append(mean_importance[feature_name] * 100.0)
                my_featname.append(feature_name)
        columns_table.append_legend()
        return columns_table, my_featname, my_prob

    def make_plt(feats, probs):
        plt.bar(range(len(feats)), probs, color='blue')
        labels = feats
        plt.xticks([x+.6 for x in range(len(feats))], labels, rotation=-70, size='small')

        plt.yticks(size='xx-small')
        plt.ylabel('Probability Feature in a Decision Tree (%)')
        plt.xlabel('Features That Occur More Than 1 Percent of Time')
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(control.path_out_chart_b_pdf)
        plt.close()

    report = Report()
    make_header(report)
    details, my_feats, my_probs = make_details(data, control.test_months)
    make_plt(my_feats, my_probs)
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
        extra_info = []
        feature_names = Features().ege_names(control.arg.features)
        columns_table = ColumnsTable((
            ('test_month', 6, '%6s', ('test', 'month'), 'test month'),
            ('nth', 2, '%2d', (' ', 'n'), 'rank of feature (1 ==> more frequently included)'),
            ('probability', 4, '%4.1f', (' ', 'prob'), 'probability feature appears in a decision tree'),
            ('feature_name', 40, '%40s', (' ', 'feature name'), 'name of feature'),
            ),
            verbose=True)
        for test_month in test_months:
            value = data[ReductionKey(test_month)]
            if 'feature_importances' not in value.importances:
                # one month has an ensemble model
                # skip that month
                print 'chart a sees an unexpected ensemble model'
                print 'test_month', test_month
                print 'value', value
                print 'value.importance', value.importances
                print 'skipping the test month'
                print 'entering debugger'
                pdb.set_trace()
            importances = value.importances['feature_importances']
            assert value.importances['features_group'] == control.arg.features, value
            model = value.model
            assert type(model) == ResultKeyGbr or type(model) == ResultKeyRfr
            sorted_indices = importances.argsort()  # sorted first lowest, last highest
            for nth_best in xrange(n_best):
                if nth_best == len(feature_names):
                    break
                index = sorted_indices[len(importances) - nth_best - 1]
                columns_table.append_detail(
                    test_month=test_month,
                    nth=nth_best + 1,
                    probability=importances[index] * 100.0,
                    feature_name=feature_names[index]
                    )
                extra_info.append([test_month, nth_best+1, importances[index]*100.0, feature_names[index]])
            for nth in xrange(n_worst):
                break  # skip, for now
                if nth == len(feature_names):
                    break
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
        return columns_table, extra_info

    def make_plt(data, info, n_best, n_worst):
        months = (
            '200512',
            '200601', '200602', '200603', '200604', '200605', '200606',
            '200607', '200608', '200609', '200610', '200611', '200612',
            '200701', '200702', '200703', '200704', '200705', '200706',
            '200707', '200708', '200709', '200710', '200711', '200712',
            '200801', '200802', '200803', '200804', '200805', '200806',
            '200807', '200808', '200809', '200810', '200811', '200812',
            '200901', '200902',
        )
        month_range = {}
        for i in range(len(months)):
            month_range[months[i]] = i+1

        redX = []
        redY = []
        blueX = []
        blueY = []
        important_fields = (
            'LIVING SQUARE FEET',
            'LAND SQUARE FOOTAGE',
            'median_household_income',
            'fraction_owner_occupied',
            'avg_commute',)
        for i in range(len(info)):
            # pdb.set_trace()  # which check this the one field?
            if info[i][3] in important_fields:
                # OLD CODE in next line
                # if info[i][3] == 'LIVING SQUARE FEET' or info[i][3] == 'LAND SQUARE FOOTAGE' \
                # or info[i][3] == 'median_household_income' or info[i][3]=='fraction_owner_occupied'\
                # or info[i][3]=='avg_commute':
                redX.append(month_range[info[i][0]])
                redY.append(info[i][2])
            else:
                blueX.append(month_range[info[i][0]])
                blueY.append(info[i][2])
        # fig = plt.figure()
        ax = plt.subplot(111)
        ax.plot(redX, redY, 'ro', label='sw')
        ax.plot(blueX, blueY, 'bs', label='other')
        plt.ylim(0, 50)
        plt.ylabel("Probability feature in a decision tree (%)")
        plt.xlabel("Validation Month")
        plt.legend(bbox_to_anchor=(1, 1), ncol=1, fancybox=True, shadow=True)
        plt.xticks([x+.3 for x in range(1, len(month_range)+1)], months, rotation=-70, size='xx-small')
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        path = control.path_out_chart_a_pdf % (n_best, n_worst)
        plt.savefig(path)
        plt.close()

    def make_report(n_best, n_worst):
        report = Report()
        make_header(report)
        details, extra_info = make_details(data, control.test_months, n_best, n_worst)
        for line in details.iterlines():
            report.append(line)
        make_plt(data, extra_info, n_best, n_worst)
        return report

    reports = {}

    def add_report(n_best, n_worst):
        reports[(n_best, n_worst)] = make_report(n_best, n_worst)

    def len_feature_group(s):
        return len(Features().ege_names(s))

    add_report(1, 0)
    add_report(len_feature_group('s'), 0)
    add_report(len_feature_group('sw'), 0)
    add_report(len_feature_group('swp'), 0)
    add_report(len_feature_group('swpn'), 0)
    return reports  # for now, skip n_worst reports
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
    'return the reduction dictionary'
    result = {}
    for test_month in control.test_months:
        path = '%s%s.pickle' % (
            control.path_in_valavm_dir,
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
                    rmse, mae, ci95_low, ci95_high = errors.errors(actuals, predictions)
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
            key = ReductionKey(
                test_month=test_month)
            value = ReductionValue(
                model=best_key,
                importances=best_importances,
                mae=lowest_mae,
                )
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
