'''reduce all the fit-predict output into a single large CSV file with all predictions

INVOCATION
  python fit-predict-reduce.py [--test] [--trace] [--cache]

where
 --cache means to cache the reading of the training and testing data from samples2

EXAMPLES OF INVOCATIONS
 python fit-predict.py train-global en 200701   # fit on training data global en models and predict Jan 2007
 python fit-predict.py all-MALIBU gb 200903      # fit on train + test data using just MALIBU data

INPUTS
 WORKING/samples2/train.csv
 WORKING/samples2/all.csv
 WORKING/fit-predict/{training_data}-{neighborhood}-{model}-{prediction_month}/{hps}.pickle
 WORKING/fit-predict/{training_data}-{neighborhood}-{model}-{prediction_month}/transaction_ids.pickle
 WORKING/fit-predict/{training_data}-{neighborhood}-{model}-{prediction_month}/actuals.pickle

OUTPUTS
 WORKING/fit-predict-reduce/reduction.csv with columns
   transaction_id apn sale_date
   training_data in {all, train}
   query_in_testing_set query_in_training_set
   model hps_str hps_1 ... (all hyperparameters)
   model_neighborhood
   price_actual price_predicted
   model_coef model_intercept model_importances
'''

from __future__ import division

import argparse
import cPickle as pickle
import datetime
import gc
import os
import pandas as pd
import pdb
from pprint import pprint
import random
import sklearn
import sklearn.ensemble
import sklearn.linear_model
import sys
import time

from Bunch import Bunch
from Cache import Cache
import dirutility
from Features import Features
import HPs
import layout_transactions
from Logger import Logger
from lower_priority import lower_priority
from Month import Month
from Path import Path
from Timer import Timer
from TransactionId import TransactionId


def make_control(argv):
    'return a Bunch'
    def neighborhood_type(s):
        return (
            s if s == 'global' else
            s.replace('_', ' ')
        )

    def month_type(s):
        try:
            Month(s)
            return s
        except:
            raise argparse.ArgumentTypeError('%s is not a valid month (YYYYMM)' % s)

    print argv
    parser = argparse.ArgumentParser()
    parser.add_argument('invocation')
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--dry', action='store_true')     # don't write output
    arg = parser.parse_args(argv)
    arg.me = arg.invocation.split('.')[0]

    if arg.trace:
        pdb.set_trace()

    random_seed = 123
    random.seed(random_seed)

    dir_working = Path().dir_working()
    dir_out = (
        os.path.join(dir_working, arg.me + '-test') if arg.test else
        os.path.join(dir_working, arg.me)
    )
    dirutility.assure_exists(dir_out)

    return Bunch(
        arg=arg,
        path_cache=os.path.join(dir_out, 'cache.pickle'),
        path_in_dir_fit_predict=os.path.join(dir_working, 'fit-predict-v2', ''),  # TODO: remove v2
        path_in_query_samples_all=os.path.join(dir_working, 'samples2', 'all.csv'),
        path_in_query_samples_train=os.path.join(dir_working, 'samples2', 'train.csv'),
        path_out_csv=os.path.join(dir_out, 'reduction.csv'),
        path_out_log=os.path.join(dir_out, '0log.txt'),
        random_seed=random_seed,
        timer=Timer(),
    )


class FittingError(Exception):
    def __init__(self, message):
        super(FittingError, self).__init__(message)


def select_in_time_period(df, last_month_str, n_months_back):
    'return subset of DataFrame df that are in the time period'
    first_date_float = float(Month(last_month_str).decrement(n_months_back).as_int() * 100 + 1)
    next_month = Month(last_month_str).increment()
    last_date = datetime.date(next_month.year, next_month.month, 1) - datetime.timedelta(1)
    last_date_float = last_date.year * 10000.0 + last_date.month * 100.0 + last_date.day

    sale_date_column = layout_transactions.sale_date
    sale_dates_float = df[sale_date_column]  # type is float
    assert isinstance(sale_dates_float.iloc[0], float)

    mask1 = sale_dates_float >= first_date_float
    mask2 = sale_dates_float <= last_date_float
    mask_in_range = mask1 & mask2
    df_in_range = df.loc[mask_in_range]
    return df_in_range


def select_in_city(df, city):
    'return subset of DataFrame df that are in the city'
    pdb.set_trace()
    cities = df[layout_transactions.city]
    mask = cities == city
    df_in_city = df.loc[mask]
    return df_in_city


def select_in_time_period_and_in_city(df, last_month, n_months_back, neighborhood):
    'return new df with the specified training data'
    verbose = False
    in_time_period = select_in_time_period(
        df.copy(),
        last_month,
        n_months_back,
    )
    in_neighborhood = (
        in_time_period if neighborhood == 'global' else
        select_in_city(in_time_period, neighborhood)
    )
    if verbose:
        print 'neighborhood %s: %d in time period, %d also in neighborhood' % (
            'all' if neighborhood is None else neighborhood,
            len(in_time_period),
            len(in_neighborhood),
        )
    return in_neighborhood


def fit_en(X, y, hps, random_seed):
    'return fitted ElastNet model'
    assert len(hps) == 5
    model = sklearn.linear_model.ElasticNet(
        alpha=hps['alpha'],
        l1_ratio=hps['l1_ratio'],
        random_state=random_seed,
        # all these parameters are at the default value format skikit-learn version 0.18.1
        fit_intercept=True,
        normalize=False,
        max_iter=1000,
        copy_X=False,
        tol=0.0001,
        warm_start=False,
        selection='cyclic',
    )
    fitted = model.fit(X, y)
    return fitted


def fit_gb(X, y, hps, random_seed):
    'return fitted GradientBoostingRegressor model'
    pdb.set_trace()
    assert len(hps) == 7
    model = sklearn.ensemble.GradientBoostingRegressor(
        learning_rate=hps['learning_rate'],
        n_estimators=hps['n_estimators'],
        max_depth=hps['max_depth'],
        max_features=hps['max_features'],
        random_state=random_seed,
        # all these parameters are at the default value format skikit-learn version 0.18.1
        loss='ls',
        criterion='friedman_mse',
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0,
        subsample=1.0,
        max_leaf_nodes=None,
        min_impurity_split=1e-7,
        alpha=0.9,
        init=None,
        verbose=0,
        presort='auto',
    )
    fitted = model.fit(X, y)
    return fitted


def fit_rf(X, y, hps, random_seed):
    'return fitted RandomForestRegressor model'
    assert len(hps) == 6
    model = sklearn.ensemble.RandomForestRegressor(
        n_estimators=hps['n_estimators'],
        max_features=hps['max_features'],
        max_depth=hps['max_depth'],
        random_state=random_seed,
        # all these parameters are at the default value format skikit-learn version 0.18.1
        criterion='mse',
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0,
        max_leaf_nodes=None,
        min_impurity_split=1e-7,
        bootstrap=True,
        oob_score=False,
        n_jobs=1,
        verbose=0,
        warm_start=False,
    )
    fitted = model.fit(X, y)
    return fitted


def make_n_hps(model):
    'return number of hyperparameters'
    count = 0
    for hps in HPs.iter_hps_model(model):
        count += 1
    return count


def fit_and_predict(training_samples, query_samples, hps, control):
    'return (predictions, attributes, n_training_samples)'
    def X_y(df):
        return Features().extract_and_transform(df, hps['units_X'], hps['units_y'])

    relevant_training_samples = select_in_time_period_and_in_city(
        training_samples,
        Month(control.arg.prediction_month).decrement(1),
        hps['n_months_back'],
        control.arg.neighborhood,
    )
    if len(relevant_training_samples) == 0:
        message = 'no relevant samples hps:%s neighborhood: %s prediction_month %s' % (
            HPs.to_str(hps),
            control.arg.neighborhood,
            control.arg.prediction_month,
        )
        raise FittingError(message)

    X_train, y_train = X_y(relevant_training_samples)
    X_query, actuals = X_y(query_samples)

    fitter = (
        fit_en if control.arg.model == 'en' else
        fit_gb if control.arg.model == 'gb' else
        fit_rf
    )
    fitted = fitter(X_train, y_train, hps, control.random_seed)
    attributes = (
        {'coef_': fitted.coef_, 'intercept_': fitted.intercept_} if control.arg.model == 'en' else
        {'feature_importances_': fitted.feature_importances_}
    )
    predictions = fitted.predict(X_query)
    return predictions, attributes, len(relevant_training_samples)


def do_work(control):
    'create csv file that summarizes all actual and predicted prices'
    def read_csv(path):
        df = pd.read_csv(
            path,
            nrows=4000 if control.arg.test else None,
            usecols=None,  # TODO: change to columns we actually use
            low_memory=False
        )
        print 'read %d samples from file %s' % (len(df), path)
        return df

    def make_transaction_ids(df):
        'return dates and apns for the query samples'
        result = []
        for index, row in df.iterrows():
            next = TransactionId(
                sale_date=row[layout_transactions.sale_date],
                apn=row[layout_transactions.apn],
            )
            result.append(next)
        return result

    def transaction_id_set(path):
        'return set of TransactionId for file at the path'
        df = read_csv(path)
        lst = make_transaction_ids(df)
        result = set(lst)
        assert len(lst) == len(result), path
        return result

    def make_testing_training_sets(control):
        pdb.set_trace()
        all = transaction_id_set(control.path_in_query_samples_all)
        train = transaction_id_set(control.path_in_query_samples_train)
        query_in_testing_set = set()
        query_in_training_set = set()
        for query in all:
            if query in train:
                query_in_training_set.add(query)
            else:
                query_in_testing_set.add(query)
        return query_in_testing_set, query_in_training_set

    def make_testing_training_setsOLD(control):
        'return (query_in_testing_set, query_in_training_set)'
        pdb.set_trace()
        return read_samples(control)
        # TODO: delete obsolete code
        if use_cache:
            start_time = time.time()
            if os.path.exists(control.path_cache):
                with open(control.path_cache, 'r') as f:
                    cache = pickle.load(f)
                print 'read cache; elapsed wall clock time', time.time() - start_time
            else:
                cache = make_cache()
                print 'create cache; elapsed wall clock time', time.time() - start_time
                start_time = time.time()
                with open(control.path_cache, 'w') as f:
                    pickle.dump(cache, f)
                print 'write cache: elapsed wall clock time', time.time() - start_time
        else:
            start_time = time.time()
            cache = make_cache()
            print 'make_cache: elapsed wall clock time', time.time() - start_time
        return cache

    def make_rows(dirname, hps_str, actuals, feature_names, transaction_ids, predictions, fitted_attributes):
        'for now, return pd.DataFrame; later return several DataFrames'
        def in_prediction_month(sale_date, prediction_month):
            factor_year = 10000.0
            factor_month = 100.0
            sale_date_year = int(sale_date / factor_year)
            sale_date_month = int((sale_date - sale_date_year * factor_year) / factor_month)
            return sale_date_year == prediction_month.year and sale_date_month == prediction_month.month

        print 'dirname', dirname
        training_data, neighborhood, model, prediction_month_str = dirname.split('-')
        prediction_month = Month(prediction_month_str)
        hps = HPs.from_str(hps_str)
        result = pd.DataFrame()
        for i in xrange(len(actuals)):
            sale_date = transaction_ids[i].sale_date
            if not in_prediction_month(sale_date, prediction_month):
                # print 'skipping: %s not in %s' % (sale_date, prediction_month)
                continue
            print 'saving', transaction_ids[i]
            row = {
                # transaction ID
                'transaction_id': transaction_ids[i],
                'apn': transaction_ids[i].apn,
                'sale_date': transaction_ids[i].sale_date,
                # what model was trained
                'training_data': training_data,
                'neighborhood': neighborhood,
                'model': model,
                'hps_str': hps_str,
                'alpha': hps.get('alpha'),
                'l1_ratio': hps.get('l1_ratio'),
                'learning_rate': hps.get('learning_rate'),
                'max_depth': hps.get('max_depth'),
                'max_features': hps.get('max_features'),
                'n_estimators': hps.get('n_estimators'),
                'n_months_back': hps.get('n_months_back'),
                'units_X': hps.get('units_X'),
                'units_y': hps.get('units_y'),
                # prices
                'actual': actuals[i],
                'predictions': predictions[i],
            }
            result = result.append(row, ignore_index=True)
        pdb.set_trace()
        return result

    # determine training and testing transactions
    pdb.set_trace()
    if control.arg.cache:
        cache = Cache(verbose=True)
        query_in_testing_set, query_in_training_set = cache.read(
            make_testing_training_sets,
            control.path_cache,
            control,
        )
    else:
        query_in_testing_set, query_in_training_set = make_testing_training_sets(control)
    # query_in_testing_set, query_in_training_set = make_testing_training_set(control.arg.cache)
    print '# training queries', len(query_in_training_set)
    print '# testing queries', len(query_in_testing_set)

    # extract info from each
    result = pd.DataFrame()
    pdb.set_trace()
    for dirpath, dirnames, filenames in os.walk(control.path_in_dir_fit_predict):
        assert len(filenames) == 0, filenames
        for dirname in dirnames:
            training_data, neighborhood, model, prediction_month = dirname.split('-')
            for dirpath2, dirnames2, filenames2 in os.walk(os.path.join(dirpath, dirname)):
                assert len(dirnames2) == 0
                with open(os.path.join(dirpath2, 'actuals.pickle'), 'r') as f:
                    actuals = pickle.load(f)
                with open(os.path.join(dirpath2, 'feature_names.pickle'), 'r') as f:
                    feature_names = pickle.load(f)
                with open(os.path.join(dirpath2, 'transaction_ids.pickle'), 'r') as f:
                    transaction_ids = pickle.load(f)
                for filename2 in filenames2:
                    filepath = os.path.join(dirpath2, filename2)
                    print 'reading', filepath
                    with open(filepath, 'r') as f:
                        obj = pickle.load(f)
                        predictions = obj['predictions']
                        fitted_attributes = obj['fitted_attributes']
                        pdb.set_trace()
                        new_rows = make_rows(
                            dirname,
                            filename2[:-7],  # drop .pickle suffix
                            actuals, feature_names, transaction_ids,
                            predictions, fitted_attributes
                        )
                        print 'added %d rows from %s' % (len(result, filename2))
                        result = result.append(new_rows, ignore_index=True)
    pdb.set_trace()
    result.to_csv(control.path_out_csv)
    print 'wrote %d records to csv file' % len(result)

    # OLD BELOW ME

    # reduce process priority, to try to keep the system responsible
    lower_priority()

    with open(control.path_out_feature_names, 'w') as f:
        feature_names = Features().ege_names('swpn')
        pickle.dump(feature_names, f)

    training_samples = read_csv(control.path_in_training_samples)

    query_samples = read_csv(control.path_in_query_samples)
    with open(control.path_out_transaction_ids, 'w') as f:
        transaction_ids = make_transaction_ids(query_samples)
        pickle.dump(transaction_ids, f)
    with open(control.path_out_actuals, 'w') as f:
        X, actuals = Features().extract_and_transform(query_samples, 'natural', 'natural')
        pickle.dump(actuals, f)

    count_fitted = 0
    n_hps = make_n_hps(control.arg.model)
    for hps in HPs.iter_hps_model(control.arg.model):
        count_fitted += 1
        start_time = time.clock()  # wall clock time on Windows, processor time on Unix
        hps_str = HPs.to_str(hps)
        hp_path = os.path.join(control.path_out_dir, hps_str + ".pickle")
        if os.path.exists(hp_path):
            print 'skipped as exists: %s' % hp_path
        else:
            try:
                predictions, fitted_attributes, n_training_samples = fit_and_predict(
                    training_samples,
                    query_samples,
                    hps, control,
                )
                with open(hp_path, 'w') as f:
                    pickle.dump(
                        {'predictions': predictions, 'fitted_attributes': fitted_attributes},
                        f,
                    )
                print 'fit-predict #%4d/%4d on:%6d in: %6.2f %s %s %s %s hps: %s ' % (
                    count_fitted,
                    n_hps,
                    n_training_samples,
                    time.clock() - start_time,
                    control.arg.training_data,
                    control.arg.neighborhood,
                    control.arg.model,
                    control.arg.prediction_month,
                    hps_str,
                )
            except Exception as e:
                print 'exception: %s' % e
                pdb.set_trace()
                with (hp_path, 'w') as f:
                    pickle.dump(
                        'error in fit_and_predict: %s' % e,
                        f,
                    )

        # gc.set_debug(gc.DEBUG_STATS + gc.DEBUG_UNCOLLECTABLE)
        # collect to get memory usage stable
        # this enables multiprocessing
        unreachable = gc.collect()
        if False and unreachable != 0:
            print 'gc reports %d unreachable objects' % unreachable
            pdb.set_trace()
        if control.arg.test and count_fitted == 5:
            print 'breaking because we are testing'
            break


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(control.path_out_log)  # now print statements also write to the log file
    print control
    lap = control.timer.lap

    do_work(control)

    lap('work completed')
    if control.arg.test:
        print 'DISCARD OUTPUT: test'
    print control
    print 'done'
    return


if __name__ == '__main__':
    if False:
        # avoid pyflakes warnings
        pdb.set_trace()
        pprint()

    main(sys.argv)
