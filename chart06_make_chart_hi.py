from __future__ import division

import collections
import math
import numpy as np
import pdb

from ColumnsTable import ColumnsTable
from columns_contain import columns_contain
import errors
from Month import Month
from Report import Report
from trace_unless import trace_unless
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
            if weight < 1:
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
        for k in control.all_k_values:
            for validation_month in control.validation_months:
                h, oracle_less_best, oracle_less_ensemble = chart_h(reduction, median_prices, actuals, k, validation_month)
                hs[(k, validation_month)] = h
                comparison[(k, validation_month)] = (oracle_less_best, oracle_less_ensemble)
        # report I is in inverted order relative to chart h grouped_by
        # make chart i part 1
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
        for city in reduction.keys():
            city_reduction = reduction[city]
            city_median_prices = median_prices[city]
            city_actuals = actuals[city]
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
