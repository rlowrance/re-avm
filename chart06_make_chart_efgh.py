from __future__ import division

import math
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pdb

from Bunch import Bunch
from ColumnsTable import ColumnsTable
from columns_contain import columns_contain
import errors
from Month import Month
from Report import Report
cc = columns_contain


class ChartEReport(object):
    def __init__(self, k, ensemble_weighting, column_definitions, test):
        self._column_definitions = column_definitions
        self._test = test
        self._report = Report()
        self._header(k, ensemble_weighting)
        cd = self._column_definitions.defs_for_columns(
            'validation_month', 'model', 'n_months_back',
            'n_estimators', 'max_features', 'max_depth',
            'learning_rate', 'rank', 'weight',
            'mae_validation', 'mae_query', 'mae_ensemble',
        )
        self._ct = ColumnsTable(columns=cd, verbose=True)

    def write(self, path):
        self._ct.append_legend()
        for line in self._ct.iterlines():
            self._report.append(line)
        if self._test:
            self._report.append('** TESTING: DISCARD')
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
    def __init__(self, k, ensemble_weighting, column_definitions, test):
        self._column_definitions = column_definitions
        self._test = test
        self._report = Report()
        self._header(k, ensemble_weighting)
        cd = self._column_definitions.defs_for_columns(
            'validation_month',
            'mae_index0',
            'mae_ensemble',
            'mae_best_next_month',
            'median_price',
            'fraction_median_price_next_month_index0',
            'fraction_median_price_next_month_ensemble',
            'fraction_median_price_next_month_best',
        )
        self._ct = ColumnsTable(columns=cd, verbose=True)

    def write(self, path):
        self._ct.append_legend()
        for line in self._ct.iterlines():
            self._report.append(line)
        if self._test:
            self._report.append('** TESTING: DISCARD')
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


def trace_unless(condition, message, **kwds):
    'like assert condition, message; but enters debugger if condition fails'
    if condition:
        return
    print '+++++++++++++++'
    for k, v in kwds.iteritems():
        print k, v
    print message
    print '+++++++++++++++'
    pdb.set_trace()


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


# return string describing key features of the model
def short_model_description(model_description):
    # build model decsription
    model = model_description.model
    if model == 'gb':
        description = '%s(%d, %d, %s, %d, %3.2f)' % (
            model,
            model_description.n_months_back,
            model_description.n_estimators,
            model_description.max_features,
            model_description.max_depth,
            model_description.learning_rate,
        )
    elif model == 'rf':
        description = '%s(%d, %d, %s, %d)' % (
            model,
            model_description.n_months_back,
            model_description.n_estimators,
            model_description.max_features,
            model_description.max_depth,
        )
    else:
        assert model == 'en', model_description
        description = '%s(%f, %f)' % (
            model,
            model_description.alpha,
            model_description.l1_ratio,
        )
    return description


class ChartHReport(object):
    def __init__(self, k, validation_month, ensemble_weighting, column_definitions, test):
        self._column_definitions = column_definitions
        self._report = Report()
        self._test = test
        self._header(k, validation_month, ensemble_weighting)
        cd = self._column_definitions.defs_for_columns(
            'description',
            'mae_validation',
            'mae_query',
            'mare_validation',
            'mare_query',
        )
        self._ct = ColumnsTable(columns=cd, verbose=True)

    def write(self, path):
        self._ct.append_legend()
        for line in self._ct.iterlines():
            self._report.append(line)
        if self._test:
            self._report.append('** TESTING: DISCARD')
        self._report.write(path)

    def detail_line(self, **kwds):
        with_spaces = {
            k: (None if self._column_definitions.replace_by_spaces(k, v) else v)
            for k, v in kwds.iteritems()
        }
        self._ct.append_detail(**with_spaces)

    def preformatted_line(self, line):
        print line
        self._ct.append_line(line)

    def _header(self, k, validation_month, ensemble_weighting):
        self._report.append('Performance of Best Models Separately and as an Ensemble')
        self._report.append(' ')
        self._report.append('Considering Best K = %d models' % k)
        self._report.append('For validation month %s' % validation_month)
        self._report.append('Ensemble weighting: %s' % ensemble_weighting)


def make_chart_efh(k, reduction, actuals, median_price, control):
    '''Write charts e and f, return median-absolute-relative_regret object'''
    def interesting():
        return k == 5

    def trace_if_interesting():
        if interesting():
            print 'k', k
            pdb.set_trace()
            return True
        else:
            return False

    ensemble_weighting = 'exp(-MAE/100000)'
    mae = {}
    debug = False
    my_validation_months = []
    my_ensemble_mae = []
    my_best_mae = []
    my_price = []
    for validation_month in control.validation_months:
        e = ChartEReport(k, ensemble_weighting, control.column_definitions, control.test)
        h = ChartHReport(k, ensemble_weighting, control.column_definitions, control.test)
        if debug:
            print validation_month
            pdb.set_trace()
        query_month = Month(validation_month).increment(1).as_str()
        if query_month not in reduction:
            control.exceptions.append('%s not in reduction (charts ef)' % query_month)
            print control.exception
            continue
        cum_weighted_predictions = None
        cum_weights = 0
        mae_validation = None
        check_key_order(reduction[validation_month])
        # write lines for the k best individual models
        # accumulate info needed to build the ensemble model
        index0_mae = None
        for index, query_month_key in enumerate(reduction[query_month].keys()):
            # print only k rows
            if index >= k:
                break
            print index, query_month_key
            validation_month_value = reduction[validation_month][query_month_key]
            print query_month
            query_month_value = reduction[query_month][query_month_key]
            if mae_validation is not None and False:  # turn off this test for now
                trace_unless(mae_validation <= validation_month_value.mae,
                             'should be non-decreasing',
                             mae_previous=mae_validation,
                             mae_next=validation_month_value.mae,
                             )
            mae_validation = validation_month_value.mae

            mae_query = query_month_value.mae
            if index == 0:
                index0_mae = mae_query
            eta = 1.0
            weight = math.exp(-eta * (mae_validation / 100000.0))
            e.detail_line(
                validation_month=validation_month,
                model=query_month_key.model,
                n_months_back=query_month_key.n_months_back,
                n_estimators=query_month_key.n_estimators,
                max_features=query_month_key.max_features,
                max_depth=query_month_key.max_depth,
                learning_rate=query_month_key.learning_rate,
                rank=index + 1,
                mae_validation=mae_validation,
                weight=weight,
                mae_query=mae_query,
            )

            h.detail_line(
                validation_month=validation_month,
                model_description=short_model_description(query_month_key),
                mae_validation=mae_validation,
                mae_query=mae_query,
            )
            # need the mae of the ensemble
            # need the actuals and predictions? or is this already computed
            predictions_next = query_month_value.predictions
            if cum_weighted_predictions is None:
                cum_weighted_predictions = weight * predictions_next
            else:
                cum_weighted_predictions += weight * predictions_next
            cum_weights += weight
        # write line comparing the best individual model in the next month
        # to the ensemble model
        trace_if_interesting()
        ensemble_predictions = cum_weighted_predictions / cum_weights
        ensemble_rmse, ensemble_mae, ensemble_ci95_low, ensemble_ci95_high = errors.errors(
            actuals[query_month],
            ensemble_predictions,
        )
        best_key = reduction[query_month].keys()[0]
        best_value = reduction[query_month][best_key]
        e.detail_line(
            validation_month=validation_month,
            mae_ensemble=ensemble_mae,
            model=best_key.model,
            n_months_back=best_key.n_months_back,
            n_estimators=best_key.n_estimators,
            max_features=best_key.max_features,
            max_depth=best_key.max_depth,
            learning_rate=best_key.learning_rate,
        )
        h.detail_line(
            validation_month=validation_month,
            model_description='ensemble',
            mae_query=ensemble_mae,
            )
        my_validation_months.append(validation_month)
        my_ensemble_mae.append(ensemble_mae)
        my_best_mae.append(best_value.mae)

        e.write(control.path_out_e_txt % (k, validation_month))
        mae[validation_month] = Bunch(
            index0=index0_mae,
            ensemble=ensemble_mae,
            best_next_month=best_value.mae,
        )

    my_ensemble_mae = []
    my_best_mae = []
    my_price = []
    for month in my_validation_months:
        my_ensemble_mae.append(mae[month].ensemble)
        my_best_mae.append(mae[month].best_next_month)
        my_price.append(median_price[Month(month)])

    width = 0.35

    fig = plt.figure()
    fig1 = fig.add_subplot(211)
    fig1.bar(
        [x+width for x in range(len(my_validation_months))],
        my_best_mae,
        width,
        color='white',
        )
    fig1.bar(
        range(len(my_validation_months)),
        my_ensemble_mae,
        width,
        color='black',
        )

    plt.ylim(0, 180000)

    labels = my_validation_months
    plt.xticks(
        [x+.4 for x in range(len(my_validation_months))],
        labels,
        rotation=-70,
        size='xx-small',
        )

    plt.ylabel('MAE ($)')
    plt.xlabel('Year-Month')

    white_patch = mpatches.Patch(
        facecolor='white',
        edgecolor='black',
        hatch='',
        lw=1,
        label="MAE of Best Model in Validation Month",
        )
    black_patch = mpatches.Patch(
        facecolor='black',
        edgecolor='black',
        hatch='',
        lw=1,
        label="MAE of Ensemble of " + str(k) + " Best Models in Validation Month",
        )
    plt.legend(handles=[white_patch, black_patch], loc=2)

    fig2 = fig.add_subplot(212)

    fig2.bar(
        [x+width for x in range(len(my_validation_months))],
        [int(m) / int(p) for m, p in zip(my_best_mae, my_price)],
        width,
        color='white',
        )
    fig2.bar(
        range(len(my_validation_months)),
        [int(m) / int(p) for m, p in zip(my_ensemble_mae, my_price)],
        width,
        color='black',
        )
    plt.ylim(0, .5)
    labels = my_validation_months
    plt.xticks(
        [x+.4 for x in range(len(my_validation_months))],
        labels,
        rotation=-70,
        size='xx-small',
        )

    plt.ylabel('Absolute Relative Error')
    plt.xlabel('Year-Month')

    white_patch = mpatches.Patch(
        facecolor='white',
        edgecolor='black',
        hatch='',
        lw=1,
        label="ARE of Best Model in Validation Month",
        )
    black_patch = mpatches.Patch(
        facecolor='black',
        edgecolor='black',
        hatch='',
        lw=1,
        label="ARE of Ensemble of " + str(k) + " Best Models in Validation Month",
        )
    plt.legend(handles=[white_patch, black_patch], loc=2)

    plt.tight_layout(pad=0.8, w_pad=0.8, h_pad=1.0)
    plt.savefig(control.path_out_e_pdf % k)

    plt.close()

    f = ChartFReport(k, ensemble_weighting, control.column_definitions, control.test)
    regrets = []
    relative_errors = []
    for validation_month in control.validation_months:
        query_month = Month(validation_month).increment(1).as_str()
        print query_month
        print "need to define best_next_month  --> best_query_month"
        pdb.set_trace()
        query_month_value = reduction[query_month][query_month_key]
        regret = mae[validation_month].ensemble - mae[validation_month].best_next_month
        regrets.append(regret)
        relative_error = regret / median_price[Month(validation_month)]
        relative_errors.append(relative_error)
        median_price_next = median_price[Month(query_month)]
        f.detail_line(
            validation_month=validation_month,
            mae_index0=mae[validation_month].index0,
            mae_ensemble=mae[validation_month].ensemble,
            mae_best_next_month=mae[validation_month].best_next_month,
            median_price=median_price[Month(validation_month)],
            fraction_median_price_next_month_index0=mae[validation_month].index0 / median_price_next,
            fraction_median_price_next_month_ensemble=mae[validation_month].ensemble / median_price_next,
            fraction_median_price_next_month_best=mae[validation_month].best_next_month / median_price_next,
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


def make_chart_efgh(reduction, actuals, median_prices, control):
    # chart g uses the regret values that are computed in building chart e
    debug = True
    g = ChartGReport()
    ks = control.all_k_values
    if control.test:
        ks = (1, 5)
    for k in ks:
        median_absolute_relative_regret = make_chart_efh(k, reduction, actuals, median_prices, control)
        if not debug:
            g.detail(k, median_absolute_relative_regret)
    if not debug:
        g.write(control.path_out_g)
