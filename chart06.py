'''create charts showing results of valgbr.py
INVOCATION
  python chart06.py FEATURESGROUP-HPS-LOCALITY --data
  python chart06.py FEATURESGROUP-HPS-global [--test] [--subset] [--norwalk] [--all]
  python chart06.py FEATURESGROUP-HPS-city [--test] [--subset] [--norwalk] [--all]  [--trace]
where
  FEATURESGROUP is one of {s, sw, swp, swpn}
  HPS is one of {all, best1}
  LOCALILTY is one of {city, global}
  FHL is FEATURESGROUP-HPS-LOCALITY
  --test means to set control.arg.test to True
  --subset means to process 0data-subset, not 0data, the full reduction
  --norwalk means to process 0data-norwalk, not 0data, the full reduction
  --all means to process all the cities, not just selected cities
  --trace start with pdb.set_trace() call, so that we run under the debugger
INPUT FILES
 WORKING/chart01/data.pickle
 WORKING/valavm/FHL/YYYYMM.pickle
 WORKING/samples-train-analysis/transactions.csv  has transaction IDs
INPUT AND OUTPUT FILES (build with --data)
 WORKING/chart06/FHL/0data.pickle         | reduction for everything
 WORKING/chart06/FHL/0data-norwalk.pickle | reduction for just Norwalk (for testing); only if locality == city
 WORKING/chart06/FHL/0data-subset.pickle | random subset of everything (for testing)
 WORKING/chart06/FHL/0all-price-histories.pickle |  
OUTPUT FILES
 WORKING/chart06/FHL/0data-report.txt | records retained TODO: Decide whether to keep
 WORKING/chart06/FHL/a.pdf           | range of losses by model (graph)
 WORKING/chart06/FHL/b-YYYYMM.pdf    | HPs with lowest losses
 WORKING/chart06/FHL/b-YYYYMM.txt    | HPs with lowest losses
 WORKING/chart06/FHL/c.pdf           | best model each month
 WORKING/chart06/FHL/d.pdf           | best & 50th best each month
 WORKING/chart06/FHL/e.pdf           | best 50 models each month (was chart07)
 WORKING/chart06/FHL/best.pickle     | dataframe with best choices each month # CHECK
 WORKING/chart06/FHL/h.txt
 WORKING chart06/FHL/i.txt
 WORKING chart06/FHL/i.pdf           | best value for k when L is global

The reduction is a dictionary.
- if LOCALITY is 'global', the type of the reduction is
  dict[validation_month] sd
  where sd is a sorted dictionary with type
  dict[ModelDescription] ModelResults, sorted by increasing ModelResults.mae
- if LOCALITY is 'city', the type of the reduction is
  dict[city_name] dict[validation_month] sd
'''

from __future__ import division

import argparse
import collections
import cPickle as pickle
import glob
import numpy as np
import pandas as pd
import pdb
from pprint import pprint
import random
import sys

import arg_type
from AVM import AVM
from Bunch import Bunch
from chart06_make_chart_a import make_chart_a
from chart06_make_chart_b import make_chart_b
from chart06_make_chart_cd import make_chart_cd
from chart06_make_chart_efgh import make_chart_efgh
from chart06_make_chart_hi import make_chart_hi
from chart06_types import ModelDescription, ModelResults, ColumnDefinitions
from columns_contain import columns_contain
import dirutility
import errors
from Logger import Logger
from Path import Path
from Report import Report
from Timer import Timer
from trace_unless import trace_unless
from valavmtypes import ResultKeyEn, ResultKeyGbr, ResultKeyRfr, ResultValue
cc = columns_contain


def make_control(argv):
    'return a Bunch'
    print argv
    parser = argparse.ArgumentParser()
    parser.add_argument('invocation')
    parser.add_argument('fhl', type=arg_type.features_hps_locality)
    parser.add_argument('--data', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--subset', action='store_true')
    parser.add_argument('--norwalk', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--use-samples-train-analysis-test', action='store_true')
    arg = parser.parse_args(argv)  # arg.__dict__ contains the bindings
    arg.base_name = arg.invocation.split('.')[0]

    # for now, we only know how to process global files
    # local files will probably have a different path in WORKING/valavm/
    # details to be determined
    arg.features, arg.hsheps, arg.locality = arg.fhl.split('-')
    assert arg.locality == 'global' or arg.locality == 'city', arg.fhl
    if arg.norwalk:
        assert arg.locality == 'city', argv
    if arg.trace:
        pdb.set_trace()

    random_seed = 123
    random.seed(random_seed)

    # assure output directory exists
    dir_working = Path().dir_working()
    dir_out_reduction = dirutility.assure_exists(dir_working + arg.base_name) + '/'
    dir_out = dirutility.assure_exists(dir_out_reduction + arg.fhl) + '/'

    validation_months = (
            '200612',
            '200701', '200702', '200703', '200704', '200705', '200706',
            '200707', '200708', '200709', '200710', '200711',
            )
    validation_months_long = (
            '200512',
            '200601', '200602', '200603', '200604', '200605', '200606',
            '200607', '200608', '200609', '200610', '200611', '200612',
            '200701', '200702', '200703', '200704', '200705', '200706',
            '200707', '200708', '200709', '200710', '200711', '200712',
            '200801', '200802', '200803', '200804', '200805', '200806',
            '200807', '200808', '200809', '200810', '200811', '200812',
            '200901', '200902',
            )

    def all_k_values():
        ks = range(1, 31, 1)
        ks.extend([40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200])
        return ks

    return Bunch(
        all_k_values=all_k_values(),
        arg=arg,
        column_definitions=ColumnDefinitions(),
        debug=arg.debug,
        errors=[],
        exceptions=[],
        path_in_valavm='%svalavm/%s/*.pickle' % (
            dir_working,
            arg.fhl,
            ),
        path_in_chart_01_reduction=dir_working + 'chart01/0data.pickle',
        path_in_data=dir_out + (
            '0data-subset.pickle' if arg.subset else
            '0data-norwalk.pickle' if arg.norwalk else
            '0data.pickle'
        ),
        path_in_interesting_cities=dir_working + 'interesting_cities.txt',
        path_in_transactions=(
            dir_working +
            'samples-train-analysis%s/transactions.csv' % ('-test' if arg.use_samples_train_analysis_test else '')
            ),
        path_all_price_histories=dir_out + '0all_price_histories.pickle',
        path_out_a=dir_out + 'a.pdf' if arg.locality == 'global' else dir_out + 'a-%s.pdf',
        path_out_b=dir_out + 'b-%d.txt',
        path_out_cd=dir_out + '%s.txt',
        path_out_c_pdf=dir_out+'c.pdf',
        path_out_b_pdf_subplots=dir_out + 'b.pdf',
        path_out_b_pdf=dir_out + 'b-%d.pdf',
        path_out_d=dir_out + 'd.txt',
        path_out_e_txt=dir_out + 'e-%04d-%6s.txt',
        path_out_e_pdf=dir_out + 'e-%04d.pdf',
        path_out_f=dir_out + 'f-%04d.txt',
        path_out_g=dir_out + 'g.txt',
        path_out_h_template=dir_out + ('h-%03d-%6s' if arg.locality == 'global' else 'h-%s-%03d-%6s') + '.txt',
        path_out_i_template=dir_out + ('i' if arg.locality == 'global' else 'i-%s') + '.txt',
        path_out_i_all_1_only_pdf=dir_out + 'i1-only.pdf',
        path_out_i_all_1_skip_pdf=dir_out + 'i1-skip.pdf',
        path_out_i_all_12_pdf=dir_out + 'i12-all.pdf',
        path_out_i_le_50_12_pdf=dir_out + 'i12-le50.pdf',
        path_out_data=dir_out + '0data.pickle',
        path_out_data_report=dir_out + '0data-report.txt',
        path_out_data_subset=dir_out + '0data-subset.pickle',
        path_out_data_norwalk=dir_out + '0data-norwalk.pickle',
        path_out_log=dir_out + '0log' + ('-data' if arg.data else '') + '.txt',
        random_seed=random_seed,
        sampling_rate=0.02,
        selected_cities=(
            'BEVERLY HILLS', 'CANYON COUNTRY',   # low number of transactions; high/low price
            'SHERMAN OAKS', 'POMONA',            # high number of transactions; high/low price
            'LOS ANGELES',
            ),
        test=arg.test,
        timer=Timer(),
        validation_months=validation_months,
        validation_months_long=validation_months_long,
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


def make_charts(reduction, actuals, median_prices, control):
    print 'making charts'

    make_chart_a(reduction, median_prices, control)
    make_chart_hi(reduction, actuals, median_prices, control)
    return  # charts b - g are obselete
    if control.arg.locality == 'city':
        print 'stopping charts after chart a and h, since locality is', control.arg.locality
        return
    make_chart_b(reduction, control, median_prices)

    make_chart_cd(reduction, median_prices, control, (0,), 'c')
    for n_best in (5, 100):
        report_id = 'd-%0d' % n_best
        for validation_month, month_reduction in reduction.iteritems():
            n_reductions_per_month = len(month_reduction)
            break
        detail_lines_d = range(n_best)[:n_reductions_per_month]
        make_chart_cd(reduction, median_prices, control, detail_lines_d, report_id)
    make_chart_efgh(reduction, actuals, median_prices, control)


def extract_yyyymm(path):
    file_name = path.split('/')[-1]
    base, suffix = file_name.split('.')
    yyyymm = base.split('-')[-1]
    return yyyymm


class ReductionIndex(object):
    'reduction DataFrame multiindex object'
    def __init__(self, city, validation_month, model_description):
        self.city = city
        self.validation_month = validation_month
        self.model_description = model_description

    def __hash__(self):
        return hash((self.city, self.validation_month, self.model_description))

    def __repr__(self):
        pattern = 'ReductionIndex(city=%s, validation_month=%s, model_description=%s)'
        return pattern % (self.city, self.validation_month, self.model_description)


class ReductionValue(object):
    'reduction DataFrame value object'
    def __init__(self, mae, model_results, feature_group):
        self.mae = mae
        self.model_results = model_results
        self.feature_group = feature_group

    def __hash__(self):
        return hash((self.mae, self.model_results, self.feature_group))

    def __repr__(self):
        pattern = 'ReductionValue(mae = %f, model_results=%s, feature_group=%s)'
        return pattern % (self.mae, self.model_results, self.feature_group)


def make_reduction(control):
    '''return the reduction dict'''

    def path_city(path):
        'return city in path to file'
        last = path.split('/')[-1]
        date, city = last.split('.')[0].split('-')
        return city

    def process_path(path):
        ''' return (dict, actuals, counter) for the path where
        dict has type dict[ModelDescription] ModelResult
        '''
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
            rmse, mae, low, high = errors.errors(value.actuals, value.predictions)
            result = ModelResults(
                rmse=rmse,
                mae=mae,
                ci95_low=low,
                ci95_high=high,
                predictions=value.predictions,
            )
            return result

        def update_price_history(
                price_history=None,
                ids=None,
                validation_year=None,
                validation_month=None,
                valavm_result_value=None,
                valavm_key=None,
        ):
            'return (augmented price history, counter)'
            def same_apn(df):
                return (df.apn == df.iloc[0].apn).all()

            def same_actual_price(df):
                return (df.actual_price == df.iloc[0].actual_price).all()

            def make_data_dict(key, matched):
                'return dictionary with the columns with want in the price history'
                # common columns
                assert isinstance(key, (ResultKeyEn, ResultKeyGbr, ResultKeyRfr)), (key, type(key))
                # common results (across all methods)
                result = {
                    'apn': matched.apn,
                    'year': matched.year,
                    'month': matched.month,
                    'day': matched.day,
                    'sequence_number': matched.sequence_number,
                    'date': matched.date,
                    'price_actual': price_actual,
                    'price_estimated': price_estimated,
                    'n_months_back': key.n_months_back,
                    'method': (
                        'rfr' if isinstance(key, ResultKeyRfr) else
                        'gbr' if isinstance(key, ResultKeyGbr) else
                        'en'
                        ),
                }

                # add columns appropriate for the type of result
                if isinstance(key, (ResultKeyRfr, ResultKeyGbr)):
                    result.update({
                        'n_esimators': key.n_estimators,
                        'max_features': key.max_features,
                        'max_depth': key.max_depth,
                    })
                if isinstance(key, ResultKeyGbr):
                    result.update({
                        'loss': key.loss,
                        'learning_rate': key.learning_rate,
                        })
                if isinstance(key, ResultKeyEn):
                    result.update({
                        'units_X': key.units_X,
                        'units_y': key.units_y,
                        'alpha': key.alpha,
                        'l1_ratio': key.l1_ratio,
                        })
                return result

            debug = False
            verbose = False
            if debug:
                print validation_year, validation_month,
                print len(ids)
                print ids[:5]
                print key
                print 'len(price_history)', 0 if price_history is None else len(price_history)
            assert len(valavm_result_value.actuals) == len(valavm_result_value.predictions)
            counter = collections.Counter()
            counter_key = 'year, month, price matched %d times'
            result_price_history = price_history
            for price_actual, price_estimated in zip(valavm_result_value.actuals, valavm_result_value.predictions):
                # print price_actual, price_estimated
                mask1 = ids.year == validation_year
                mask2 = ids.month == validation_month
                mask3 = ids.actual_price == price_actual
                mask = mask1 & mask2 & mask3
                if debug:
                    print 'transaction in validation year:     ', sum(mask1)
                    print 'transactions in validation month:   ', sum(mask2)
                    print 'transactions with same actual price;', sum(mask3)
                    print 'transactions that match on all:     ', sum(mask)
                    print type(key)
                matched_df = ids[mask]
                if len(matched_df) == 1:
                    counter[counter_key % 1] += 1
                    matched = matched_df.iloc[0]
                    print 'single match', matched_df.index[0]
                    data_dict = make_data_dict(key, matched)
                    df_new = pd.DataFrame(
                        data=data_dict,
                        index=[0 if result_price_history is None else len(result_price_history)],
                    )
                    # pdb.set_trace()
                    result_price_history = (
                        df_new if result_price_history is None else
                        result_price_history.append(df_new, verify_integrity=True)
                        )
                else:
                    counter[counter_key % len(matched_df)] += 1
                    if len(matched_df) > 0:
                        if verbose:
                            print 'valavm matched %d training transactions' % len(matched_df)
                            print matched_df
                    continue
                if control.debug and len(result_price_history) > 2:
                    print 'DEBUG: breaking out of update_price_history'
                    break
            print 'identified %d price histories' % len(result_price_history)
            return result_price_history, counter

        ids = pd.read_csv(
            control.path_in_transactions,
            index_col=0,
            low_memory=False,
            )
        print 'reducing', path
        model = {}
        counter = collections.Counter()
        input_record_number = 0
        actuals = None
        updated_price_history = None  # mutated in the while True loop below
        validation_year_month = int(path.split('/')[-1].split('.')[0])
        validation_year = int(validation_year_month / 100)
        validation_month = int(validation_year_month - validation_year * 100)
        assert validation_year_month == validation_year * 100 + validation_month
        with open(path, 'rb') as f:
            while True:  # process each record in path
                counter['attempted to read'] += 1
                input_record_number += 1
                if control.debug and input_record_number > 10:
                    print 'DEBUG: breaking out of record read in path', path
                    break
                try:
                    # model[model_key] = error_analysis, for next model result
                    record = pickle.load(f)
                    counter['actually read'] += 1
                    assert isinstance(record, tuple), type(record)
                    assert len(record) == 2, len(record)
                    key, value = record
                    assert len(value) == 2, len(value)
                    # NOTE: importances is not used
                    valavm_result_value, importances = value
                    # type(valavm_result_value) == namedtuple with fields actuals, predictions
                    # the fields are parallel, corresponding transaction to transaction
                    if len(ids) < 100000:
                        print 'WARNING: TRUNCATED IDS'
                    # verify that actuals is always the same
                    if actuals is not None:
                        assert np.array_equal(actuals, valavm_result_value.actuals)
                    actuals = valavm_result_value.actuals
                    # verify that each model_key occurs at most once in the validation month
                    model_key = make_model_description(key)
                    updated_price_history, update_counter = update_price_history(
                        price_history=updated_price_history,
                        ids=ids,
                        validation_year=validation_year,
                        validation_month=validation_month,
                        valavm_result_value=valavm_result_value,
                        valavm_key=key,
                    )
                    # NOTE: the counters are the same for every path because the actual transactions are the same
                    if input_record_number == 1:
                        print 'path update counters', path
                        for k, v in update_counter.iteritems():
                            print k, v
                    if model_key in model:
                        print '++++++++++++++++++++++'
                        print path, model_key
                        print 'duplicate model key'
                        pdb.set_trace()
                        print '++++++++++++++++++++++'
                    model[model_key] = make_model_result(valavm_result_value)
                except ValueError as e:
                    counter['ValueError'] += 1
                    if key is not None:
                        print key
                    print 'ignoring ValueError in record %d: %s' % (input_record_number, e)
                except EOFError:
                    counter['EOFError'] += 1
                    print 'found EOFError path in record %d: %s' % (input_record_number, path)
                    print 'continuing'
                    if input_record_number == 1 and False:
                        # with locality == city, a file can be empty
                        control.errors.append('eof record 1; path = %s' % path)
                    break
                except pickle.UnpicklingError as e:
                    counter['UnpicklingError'] += 1
                    print 'cPickle.Unpicklingerror in record %d: %s' % (input_record_number, e)

        return model, actuals, counter, updated_price_history

    reduction = collections.defaultdict(dict)
    all_actuals = collections.defaultdict(dict)
    paths = sorted(glob.glob(control.path_in_valavm))
    assert len(paths) > 0, paths
    counters = {}
    all_price_histories = None
    for path in paths:
        model, actuals, counter, price_history = process_path(path)
        all_price_histories = (
            price_history if all_price_histories is None else
            all_price_histories.append(price_history, verify_integrity=True, ignore_index=True)
            )
        # type(model) is dict[ModelDescription] ModelResults
        # sort models by increasing ModelResults.mae
        sorted_models = collections.OrderedDict(sorted(model.items(), key=lambda t: t[1].mae))
        check_key_order(sorted_models)
        if control.arg.locality == 'global':
            base_name, suffix = path.split('/')[-1].split('.')
            validation_month = base_name
            reduction[validation_month] = sorted_models
            all_actuals[validation_month] = actuals
        elif control.arg.locality == 'city':
            base_name, suffix = path.split('/')[-1].split('.')
            validation_month, city_name = base_name.split('-')
            #  some file systems create all upper case names
            #  some create mixed-case names
            #  we map each to upper case
            city_name_used = city_name.upper()
            reduction[city_name_used][validation_month] = sorted_models
            all_actuals[city_name_used][validation_month] = actuals
        else:
            print 'unexpected locality', control.arg.locality
            pdb.set_trace()
        counters[path] = counter
        if control.debug and len(counters) > 1:
            print 'DEBUG: stopping iteration over paths'
            break
        if control.test:
            break

    return reduction, all_actuals, counters, all_price_histories


def make_subset_global(reduction, fraction):
    'return a random sample of the reduction stratified by validation_month as an ordereddict'
    # use same keys (models) every validation month
    # generate candidate for common keys in the subset
    subset_common_keys = None
    for validation_month, validation_dict in reduction.iteritems():
        if len(validation_dict) == 0:
            print 'zero length validation dict', validation_month
            pdb.set_trace()
        keys = validation_dict.keys()
        n_to_keep = int(len(keys) * fraction)
        subset_common_keys_list = random.sample(keys, n_to_keep)
        subset_common_keys = set(subset_common_keys_list)
        break

    # remove keys from subset_common_keys that are not in each validation_month
    print 'n candidate common keys', len(subset_common_keys)
    for validation_month, validation_dict in reduction.iteritems():
        print 'make_subset', validation_month
        validation_keys = set(validation_dict.keys())
        for key in subset_common_keys:
            if key not in validation_keys:
                print 'not in', validation_month, ': ', key
                subset_common_keys -= set(key)
    print 'n final common keys', len(subset_common_keys)

    # build reduction subset using the actual common keys
    results = {}
    for validation_month, validation_dict in reduction.iteritems():
        d = {
            common_key: validation_dict[common_key]
            for common_key in subset_common_keys
            }
        # sort by MAE, low to high
        od = collections.OrderedDict(sorted(d.items(), key=lambda x: x[1].mae))
        results[validation_month] = od

        return results


def make_subset_city(reduction, path_interesting_cities):
    'return reduction for just the interesting cities'
    result = {}
    if len(reduction) <= 6:
        return reduction
    with open(path_interesting_cities, 'r') as f:
        lines = f.readlines()
        no_newlines = [line.rstrip('\n') for line in lines]
        for interesting_city in no_newlines:
            if interesting_city in reduction:
                result[interesting_city] = reduction[interesting_city]
            else:
                print 'not in reduction', interesting_city
                pdb.set_trace()
    return result


def make_subset(reduction, fraction, locality, interesting_cities):
    'return dict of type type(reduction) but with a randomly chosen subset of size fraction * len(reduction)'
    if locality == 'global':
        return make_subset_global(reduction, fraction)
    elif locality == 'city':
        return make_subset_city(reduction, interesting_cities)
    else:
        print 'bad locality', locality
        pdb.set_trace()


def make_norwalk(reduction):
    'return dict of type(reduction) with with just the norwalk data items'
    city = 'NORWALK'
    result = {city: reduction[city]}
    return result


def make_median_price(path, cities):
    'return dict[Month] median_price or dict[city][Month] median_price'
    def median_price(df, month):
        in_month = df.month == month
        result = df[in_month].price.median()
        return result

    with open(path, 'rb') as f:
        df, reduction_control = pickle.load(f)
        all_months = set(df.month)
        if cities:
            all_cities = set(df.city)
            result = collections.defaultdict(dict)
            for city in all_cities:
                in_city = df.city == city
                result[city] = {month: median_price(df[in_city], month) for month in all_months}
        else:
            result = {month: median_price(df, month) for month in all_months}
    return result


class ReportReduction(object):
    def __init__(self, counters):
        self._report = self._make_report(counters)

    def write(self, path):
        self._report.write(path)

    def _make_report(self, counters):
        r = Report()
        r.append('Records retained while reducing input file')
        for path, counter in counters.iteritems():
            r.append(' ')
            r.append('path %s' % path)
            for tag, value in counter.iteritems():
                r.append('%30s: %d' % (tag, value))
        return r


def main(argv):
    print "what"
    control = make_control(argv)
    sys.stdout = Logger(logfile_path=control.path_out_log)
    print control
    lap = control.timer.lap

    if control.arg.data:
        if not control.debug:
            median_price = make_median_price(control.path_in_chart_01_reduction, control.arg.locality == 'city')
            lap('make_median_price')
        reduction, all_actuals, counters, all_price_histories = make_reduction(control)
        lap('make_reduction')
        with open(control.path_all_price_histories, 'wb') as f:
            pdb.set_trace()
            print 'len(all_price_histories)', len(all_price_histories)
            print 'columns', all_price_histories.columns
            pickle.dump(all_price_histories, f)
            lap('write all price histories')
        if len(control.errors) > 0:
            print 'stopping because of errors'
            for error in control.errors:
                print error
            pdb.set_trace()
        lap('make_data')
        ReportReduction(counters).write(control.path_out_data_report)
        subset = make_subset(reduction, control.sampling_rate, control.arg.locality, control.path_in_interesting_cities)
        lap('make_subset')
        norwalk = make_norwalk(reduction) if control.arg.locality == 'city' else None
        # check key order

        def check_validation_month_keys(reduction):
            for validation_month in reduction.keys():
                check_key_order(reduction[validation_month])

        if control.arg.locality == 'global':
            check_validation_month_keys(reduction)
            check_validation_month_keys(subset)
        else:
            for city in reduction.keys():
                check_validation_month_keys(reduction[city])
            for city in subset.keys():
                check_validation_month_keys(subset[city])
        lap('check key order')

        output_all = (reduction, all_actuals, median_price, control)
        output_samples = (subset, all_actuals, median_price, control)
        output_norwalk = (norwalk, all_actuals, median_price, control)
        lap('check key order')
        with open(control.path_out_data, 'wb') as f:
            pickle.dump(output_all, f)
            lap('write all data')
        with open(control.path_out_data_subset, 'wb') as f:
            pickle.dump(output_samples, f)
            lap('write samples')
        if control.arg.locality == 'city':
            with open(control.path_out_data_norwalk, 'wb') as f:
                pickle.dump(output_norwalk, f)
                lap('write norwalk')
    else:
        with open(control.path_in_data, 'rb') as f:
            print 'reading reduction data file'
            reduction, all_actuals, median_price, reduction_control = pickle.load(f)
            lap('read input from %s' % control.path_in_data)

        # check that the reduction dictionaries are ordered by mae
        def check_order_months(d):
            for validation_month, ordered_dict in d.iteritems():
                check_key_order(ordered_dict)

        if control.arg.locality == 'global':
            check_order_months(reduction)
        elif control.arg.locality == 'city':
            for city, month_dict in reduction.iteritems():
                check_order_months(month_dict)

        make_charts(reduction, all_actuals, median_price, control)

    print control
    if control.test:
        print 'DISCARD OUTPUT: test'
    if control.debug:
        print 'DISCARD OUTPUT: debug'
    if control.arg.subset:
        print 'DISCARD OUTPUT: subset'
    if len(control.errors) != 0:
        print 'DISCARD OUTPUT: ERRORS'
        for error in control.errors:
            print error
    if len(control.exceptions) != 0:
        print 'DISCARD OUTPUT; EXCEPTIONS'
        for exception in control.expections:
            print exception
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
