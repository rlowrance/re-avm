from __future__ import division

import collections
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb

from ColumnsTable import ColumnsTable
from columns_contain import columns_contain
import errors
from Month import Month
from Report import Report
cc = columns_contain


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


class ChartIReport(object):
    def __init__(self, column_definitions, test):
        self._column_definitions = column_definitions
        self._report = Report()
        self._header()
        self._test = test
        self._appended = []
        cd = self._column_definitions.defs_for_columns(
            'validation_month',
            'k',
            'oracle_less_best',
            'oracle_less_ensemble',
            )
        self._ct = ColumnsTable(columns=cd, verbose=True)

    def write(self, path):
        self._ct.append_legend()
        for line in self._ct.iterlines():
            self._report.append(line)
        for line in self._appended:
            self._report.append(line)
        if self._test:
            self._report.append('** TESTING: DISCARD')
        self._report.write(path)

    def append(self, line):
        self._ct.append_line(line)

    def detail_line(self, **kwds):
        with_spaces = {
            k: (None if self._column_definitions.replace_by_spaces(k, v) else v)
            for k, v in kwds.iteritems()
        }
        self._ct.append_detail(**with_spaces)

    def _header(self):
        self._report.append('Performance of Best and Ensemble Models Relative to the Oracle')
        self._report.append(' ')


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


def make_confidence_intervals(df, regret_column_name):
    'return ndarrays (lower, upper) of 90% confidence intervals for each value of df.k'
    trace = True
    if trace:
        pdb.set_trace()
    n_resamples = 100  # TODO: adjust to 10,000
    lower_percentile = 10
    upper_percentile = 90
    ks = sorted(set(df.k))
    lower = np.zeros((len(ks),))
    upper = np.zeros((len(ks),))
    for i, k in enumerate(ks):
        for_k = df[df.k == k]
        values = for_k[regret_column_name]
        sample = np.random.choice(
            np.abs(values),
            size=n_resamples,
            replace=True,
            )
        lower[i] = np.percentile(sample, lower_percentile)
        upper[i] = np.percentile(sample, upper_percentile)
    if trace:
        print 'lower', lower
        print 'upper', upper
        pdb.set_trace()
    return (lower, upper)


def add_regret(df, show_confidence_interval=True):
    'mutate plt object by adding 2 regret lines'
    # TODO: remove abs
    def maybe_adjust_y_value(series):
        if sum(series == 0.0) == len(series):
            # all the values are zero and hence will plot on top of the x axis and will be invisible
            # subsitute small positive value
            max_y = np.max(np.abs(df.oracle_less_ensemble))
            fraction = (  # put the line just above the x axis
                .02 if max_y < 3000 else
                .01 if max_y < 6000 else
                .01
            )
            substitute_value = fraction * max_y
            return pd.Series([substitute_value] * len(series))
        else:
            return series

    plt.autoscale(
        enable=True,
        axis='both',
        tight=False,  # let locator and margins expand the view limits
        )
    plt.plot(
        df.k,
        np.abs(df.oracle_less_ensemble),
        'b.',  # blue point markers
        label='abs(oracle_less_ensemble)',
        )
    mean_value = np.mean(np.abs(df.oracle_less_ensemble))
    plt.plot(
        df.k,
        pd.Series([mean_value] * len(df)),
        'b-',  # blue line marker
        label='mean(abs(oracle_less_ensemble))',
    )
    if show_confidence_interval:
        # confidence interval for the blue dots (oracle_less_ensemble)
        lower, upper = make_confidence_intervals(df, 'oracle_less_ensemble')
        print 'lower', lower
        print 'upper', upper
        pdb.set_trace()
        plt.plot(
            df.k,
            lower,
            'mv',  # magenta triangle-down marker
            label='90% ci lower bound',
            )
        plt.plot(
            df.k,
            upper,
            'm^',  # magenta triangle-up market
            label='90% ci upper bound',
            )

    plt.plot(
        df.k,
        np.abs(maybe_adjust_y_value(df.oracle_less_best)),
        'r-',  # red line marker
        label='mean(abs(oracle_less_best))',
        )


def add_title(s):
    'mutate plt'
    plt.title(
        s,
        loc='right',
        fontdict={
            'fontsize': 'xx-small',
            'style': 'italic',
            },
        )


def add_labels():
    'mutate plt'
    plt.xlabel('K')
    plt.ylabel('abs(reget)')


def add_legend():
    'mutate plt'
    plt.legend(
        loc='best',
        fontsize=5,
        )


def set_layout():
    'mutate plt'
    plt.tight_layout(
        pad=0.4,
        w_pad=0.5,
        h_pad=1.0,
        )


def make_i_plt_1(df):
    'return plt, a 1-up figure with one subplot for all the validation months'
    plt.subplot(1, 1, 1)  # 1 x 1 grid, draw first subplot
    first_month = '200612'
    last_month = '200711'
    add_regret(
        df[np.logical_and(
            df.validation_month >= first_month,
            df.validation_month <= last_month)])
    add_title('yr mnth %s through yr mnth %s' % (first_month, last_month))
    add_labels()
    add_legend()
    set_layout()
    return plt


def make_i_plt_12(i_df):
    'return plt, a 12-up figure with one subplot for each validation month'
    # make the figure; imitate make_chart_a
    def make_subplot(validation_month):  # TODO: remove this dead code
        'mutate plt by adding an axes with the two regret lines for the validation_month'
        in_month = i_df[i_df.validation_month == validation_month]
        oracle_less_ensemble_x = in_month.k
        oracle_less_ensemble_y = np.abs(in_month.oracle_less_ensemble)
        plt.autoscale(enable=True, axis='both', tight=True)
        plt.plot(
            oracle_less_ensemble_x,
            oracle_less_ensemble_y,
            'b.',  # blue point markers
            label='oracle less ensemble',
        )

        oracle_less_best_x = in_month.k
        oracle_less_best_y = np.abs(in_month.oracle_less_best)  # always the same value
        if sum(oracle_less_best_y == 0.0) == len(oracle_less_best_y):
            # all the values are zero
            reset_value = 10.0  # replace 0 values with this value, so that the y value is not plotted on the x axis
            xx = pd.Series([reset_value] * len(oracle_less_best_y))
            oracle_less_best_y = xx
        plt.plot(
            oracle_less_best_x,
            oracle_less_best_y,
            'r-',  # red with solid line
            label='oracle less best',
        )
        plt.title(
            'yr mnth %s' % validation_month,
            loc='right',
            fontdict={
                'fontsize': 'xx-small',
                'style': 'italic',
                },
            )

    axes_number = 0
    validation_months = (
        '200612', '200701', '200702', '200703', '200704', '200705',
        '200706', '200707', '200708', '200709', '200710', '200711',
    )
    row_seq = (1, 2, 3, 4)
    col_seq = (1, 2, 3)
    for row in row_seq:
        for col in col_seq:
            validation_month = validation_months[axes_number]
            axes_number += 1  # count across rows
            plt.subplot(len(row_seq), len(col_seq), axes_number)
            add_regret(i_df[i_df.validation_month == validation_month])
            add_title('yr mnth %s' % validation_month)
            # make_subplot(validation_month)
        # annotate the bottom row only
        if row == 4 and col == 1:
            add_labels()
        if row == 4 and col == 3:
            add_legend()

    set_layout()
    return plt


# write report files for all K values and validation months for the year 2007
def make_chart_hi(reduction, actuals, median_prices, control):
    'return None'

    def make_dispersion_lines(report=None, tag=None, actuals=None, estimates=None):
        # append lines to report

        def quartile_median(low, hi):
            'return median error of actuals s.t. low <= actuals <= hi, return count of number of values in range'
            mask = np.array(np.logical_and(actuals >= low, actuals <= hi), dtype=bool)
            q_actuals = actuals[mask]
            if len(q_actuals) == 0:
                print 'no elements selected by mask', low, hi, sum(mask)
                return 0.0, 1.0, sum(mask)  # can't return 0.0's because if future divide by zero
            q_estimates = estimates[mask]
            q_abs_errors = np.abs(q_actuals - q_estimates)
            q_median_error = np.median(q_abs_errors)
            try:
                print 'q_actuals:', q_actuals
                q_median_value = np.percentile(q_actuals, 50)
            except Exception as e:
                pdb.set_trace()
                print type(e)
                print e.args
                print e
                pdb.set_trace()
                q_median_value = 0
            return q_median_error, q_median_value, sum(mask)

        actuals_quartiles = np.percentile(actuals, (0, 25, 50, 75, 100))

        report.preformatted_line('\nMedian Error by Price Quartile for %s\n' % tag)
        for q in (0, 1, 2, 3):
            q_median_error, q_median_value, count = quartile_median(
                actuals_quartiles[q] + (0 if q == 0 else 1),
                actuals_quartiles[q + 1] - (1 if q == 3 else 0),
            )
            report.preformatted_line('quartile %d  (prices %8.0f to %8.0f  N=%5d): median price: %8.0f median error: %8.0f error / price: %6.4f' % (
                q + 1,
                actuals_quartiles[q] + (0 if q == 0 else 1),
                actuals_quartiles[q + 1] - (1 if q == 3 else 0),
                count,
                q_median_value,
                q_median_error,
                q_median_error / q_median_value,
                ))

    def mae(actuals, predictions):
        'return named tuple'
        e = errors.errors(actuals, predictions)
        mae_index = 1
        return e[mae_index]

    def chart_h(reduction, median_prices, actuals, k, validation_month):
        'return (Report, oracle_less_best, oracle_less_ensemble)'

        def median_price(month_str):
            return median_prices[Month(month_str)]

        print 'chart_h', k, validation_month
        if k == 2 and False:
            pdb.set_trace()
        h = ChartHReport(k, validation_month, 'exp(-MAE/$100000)', control.column_definitions, control.test)
        query_month = Month(validation_month).increment(1).as_str()
        # write results for each of the k best models in the validation month
        cum_weight = None
        eta = 1.0
        weight_scale = 200000.0  # to get weight < 1
        for index in xrange(k):
            # write detail line for this expert
            try:
                expert_key = reduction[validation_month].keys()[index]
            except IndexError as e:
                h.preformatted_line('IndexError: %s' % str(e))
                h.preformatted_line('index: %d' % index)
                h.preformatted_line('giving up on completing the chart')
                return h, 1, 1
            expert_results_validation_month = reduction[validation_month][expert_key]
            if expert_key not in reduction[query_month]:
                h.preformatted_line('expert_key not in query month')
                h.preformatted_line('expert key: %s' % str(expert_key))
                h.preformatted_line('query_month: %s' % query_month)
                h.preformatted_line('index: %d' % index)
                h.preformatted_line('giving up on completing the chart')
                return h, 1, 1
            expert_results_query_month = reduction[query_month][expert_key]
            h.detail_line(
                description='expert ranked %d: %s' % (index + 1, short_model_description(expert_key)),
                mae_validation=expert_results_validation_month.mae,
                mae_query=expert_results_query_month.mae,
                mare_validation=expert_results_validation_month.mae / median_price(validation_month),
                mare_query=expert_results_query_month.mae / median_price(query_month),
                )
            # computing running ensemble model prediction
            weight = math.exp(- eta * expert_results_validation_month.mae / weight_scale)
            if not (weight < 1):
                print weight, eta, expert_results_validation_month.mae, weight_scale
                pdb.set_trace()
            assert weight < 1, (eta, expert_results_validation_month.mae, weight_scale)
            incremental_ensemble_predictions_query = weight * expert_results_query_month.predictions
            incremental_ensemble_predictions_validation = weight * expert_results_validation_month.predictions
            if cum_weight is None:
                cum_ensemble_predictions_query = incremental_ensemble_predictions_query
                cum_ensemble_predictions_validation = incremental_ensemble_predictions_validation
                cum_weight = weight
            else:
                cum_ensemble_predictions_query += incremental_ensemble_predictions_query
                cum_ensemble_predictions_validation += incremental_ensemble_predictions_validation
                cum_weight += weight
        # write detail line for the ensemble
        # pdb.set_trace()
        h.detail_line(
            description=' ',
            )
        if k == 10 and validation_month == '200705' and False:
            print k, validation_month
            pdb.set_trace()
        ensemble_predictions_query = cum_ensemble_predictions_query / cum_weight
        ensemble_predictions_validation = cum_ensemble_predictions_validation / cum_weight
        ensemble_errors_query_mae = mae(actuals[query_month], ensemble_predictions_query)
        ensemble_errors_validation_mae = mae(actuals[validation_month], ensemble_predictions_validation)
        h.detail_line(
            description='ensemble of best %d experts' % k,
            mae_validation=ensemble_errors_validation_mae,
            mae_query=ensemble_errors_query_mae,
            mare_validation=ensemble_errors_validation_mae / median_price(validation_month),
            mare_query=ensemble_errors_query_mae / median_price(query_month),
            )
        # write detail line for the oracle's model
        oracle_key = reduction[query_month].keys()[0]
        if oracle_key not in reduction[validation_month]:
            h.preformatted_line('validation month %s missing %s' % (validation_month, str(oracle_key)))
            h.preformatted_line('skipping remainder of report')
            return (h, 1.0, 1.0)
        oracle_results_validation_month = reduction[validation_month][oracle_key]
        oracle_results_query_month = reduction[query_month][oracle_key]
        h.detail_line(
            description='oracle: %s' % short_model_description(oracle_key),
            mae_validation=oracle_results_validation_month.mae,
            mae_query=oracle_results_query_month.mae,
            mare_validation=oracle_results_validation_month.mae / median_price(validation_month),
            mare_query=oracle_results_query_month.mae / median_price(query_month),
            )
        # report differences from oracle
        best_key = reduction[validation_month].keys()[0]
        best_results_query_month = reduction[query_month][best_key]
        mpquery = median_price(query_month)
        oracle_less_best_query_month = oracle_results_query_month.mae - best_results_query_month.mae
        oracle_less_ensemble_query_month = oracle_results_query_month.mae - ensemble_errors_query_mae

        def iszero(name, value):
            print name, type(value), value
            if value == 0:
                print 'zero divisor:', name, type(value), value
                return True
            else:
                return False

        h.detail_line(
            description=' ',
            )
        h.detail_line(
            description='oracle - expert ranked 1',
            mae_query=oracle_less_best_query_month,
            mare_query=oracle_results_query_month.mae / mpquery - best_results_query_month.mae / mpquery,
            )
        h.detail_line(
            description='oracle - ensemble model',
            mae_query=oracle_less_ensemble_query_month,
            mare_query=oracle_results_query_month.mae / mpquery - ensemble_errors_query_mae / mpquery,
            )
        h.detail_line(
            description=' ',
            )
        if oracle_results_query_month.mae == 0.0:
            h.detail_line(description='relative regrets are infinite because oracle MAE is 0')
            h.detail_line(
                description='100*(oracle - expert ranked 1)/oracle',
            )
            h.detail_line(
                description='100*(oracle - ensemble model)/oracle',
            )
        else:
            h.detail_line(
                description='100*(oracle - expert ranked 1)/oracle',
                mae_query=100 * (oracle_less_best_query_month / oracle_results_query_month.mae),
            )
            h.detail_line(
                description='100*(oracle - ensemble model)/oracle',
                mae_query=100 * (oracle_less_ensemble_query_month / oracle_results_query_month.mae),
            )
        # dispersion of errors relative to prices
        make_dispersion_lines(
            report=h,
            tag='ensemble',
            actuals=actuals[query_month],
            estimates=ensemble_predictions_query,
            )
        return h, oracle_less_best_query_month, oracle_less_ensemble_query_month

    def median_value(value_list):
        sum = 0.0
        for value in value_list:
            sum += value
        return sum / len(value_list)

    def make_hi(reduction, median_prices, actuals):
        'return (dict[(k, validation_month)]Report, Report)'
        # make chart h
        hs = {}
        comparison = {}
        i_df = None
        for k in control.all_k_values:
            for validation_month in control.validation_months:
                h, oracle_less_best, oracle_less_ensemble = chart_h(reduction, median_prices, actuals, k, validation_month)
                hs[(k, validation_month)] = h
                comparison[(k, validation_month)] = (oracle_less_best, oracle_less_ensemble)
                new_i_df = pd.DataFrame(
                    data={
                        'k': k,
                        'validation_month': validation_month,
                        'oracle_less_best': oracle_less_best,
                        'oracle_less_ensemble': oracle_less_ensemble,
                    },
                    index=['%03d-%s' % (k, validation_month)],
                )
                i_df = new_i_df if i_df is None else i_df.append(new_i_df, verify_integrity=True)
        # report I is in inverted order relative to chart h grouped_by
        # make graphical report to help select the best value of k
        if control.arg.locality == 'global':
            def write_i_plot_12(df, path):
                i_plt = make_i_plt_12(df)
                i_plt.savefig(path)
                i_plt.close()

            def write_i_plot_1(df, path):
                # replace df.oralce_less_best with it's mean value
                copied = df.copy()
                new_value = np.mean(df.oracle_less_best)
                copied.oracle_less_best = pd.Series([new_value] * len(df), index=df.index)
                i_plt = make_i_plt_1(copied)
                i_plt.savefig(path)
                i_plt.close()

            write_i_plot_1(i_df, control.path_out_i_all_1_pdf)
            write_i_plot_12(i_df, control.path_out_i_all_12_pdf)
            write_i_plot_12(i_df[i_df.k <= 50], control.path_out_i_le_50_12_pdf)

        # create text report (this can be deleted later)
        i = ChartIReport(control.column_definitions, control.test)
        count = 0
        sum_abs_oracle_less_best = 0
        sum_abs_oracle_less_ensemble = 0
        oracle_less_ensemble_by_k = collections.defaultdict(list)
        for validation_month in control.validation_months:
            for k in control.all_k_values:
                oracle_less_best, oracle_less_ensemble = comparison[(k, validation_month)]
                i.detail_line(
                    validation_month=validation_month,
                    k=k,
                    oracle_less_best=oracle_less_best,
                    oracle_less_ensemble=oracle_less_ensemble,
                )
                oracle_less_ensemble_by_k[k].append(oracle_less_ensemble)
                count += 1
                sum_abs_oracle_less_best += abs(oracle_less_best)
                sum_abs_oracle_less_ensemble += abs(oracle_less_ensemble)

        # make chart i part 2 (TODO: create separate chart)
        i.append(' ')
        i.append('Median (oracle - ensemble)')
        for k in sorted(oracle_less_ensemble_by_k.keys()):
            value_list = oracle_less_ensemble_by_k[k]
            i.detail_line(
                k=k,
                oracle_less_ensemble=median_value(value_list),
                )
        i.append(' ')
        i.append('median absolute oracle less best    : %f' % (sum_abs_oracle_less_best / count))
        i.append('median absolute oracle less ensemble: %f' % (sum_abs_oracle_less_ensemble / count))
        return hs, i

    control.timer.lap('start charts h and i')
    if control.arg.locality == 'global':
        hs, i = make_hi(reduction, median_prices, actuals)
        # write the reports (the order of writing does not matter)
        for key, report in hs.iteritems():
            k, validation_month = key
            report.write(control.path_out_h_template % (k, validation_month))
        i.write(control.path_out_i_template)
        return
    elif control.arg.locality == 'city':
        cities = reduction.keys() if control.arg.all else control.selected_cities
        for city in cities:
            city_reduction = reduction[city]
            city_median_prices = median_prices[city]
            city_actuals = actuals[city]
            print city, len(city_reduction), city_median_prices
            print 'city:', city
            hs, i = make_hi(city_reduction, city_median_prices, city_actuals)
            # write the reports (the order of writing does not matter)
            if hs is None:
                print 'no h report for city', city
                continue
            for key, report in hs.iteritems():
                k, validation_month = key
                report.write(control.path_out_h_template % (city, k, validation_month))
            i.write(control.path_out_i_template % city)
        return
    else:
        print control.arg.locality
        print 'bad locality'
        pdb.set_trace()
