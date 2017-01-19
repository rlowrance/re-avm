'''fit and predict with many models

INVOCATION
  python fit-predict.py {training_data} {neighborhood} {model} {prediction_month}

where
 training_data     in {train, all} specifies which data in Working/samples2 to use
 neighborhood      in {global, city_name} specifies whether to train a model on all cities or just the specified city
 model             in {en, gb, rf} specified which model to use
 prediction_month  like YYYYMM specfies the month for which all samples are predicted

EXAMPLES OF INVOCATIONS
 python fit-predict.py train-global en 200701   # fit on training data global en models and predict Jan 2007
 python fit-predict.py all-MALIBU gb 200903      # fit on train + test data using just MALIBU data

INPUTS
 WORKING/samples2/train.csv or
 WORKING/samples2/all.csv

OUTPUTS

  WORKING/fit[-test]/{training-data}-{neighborhood}-{model}-{prediction_month}/feature_names.pickle
   A tuple with X feature names in order (use to decode the fitted_attributes)
  WORKING/fit[-test]/{training-data}-{neighborhood}-{model}-{prediction_month}/transaction_ids.pickle
   A list of TransactionId(apn, date) of the query transactions
 WORKING/fit[-test]/{training-data}-{neighborhood}-{model}-{prediction_month}/actuals.pickle
   A numpy 1D array with the actual prices, parallel to the transaction_ids

 WORKING/fit[-test]/{training_data}-{neighborhood}-{model}-{prediction_month}/predictions-attributes.pickle
   A file with pickled tuples (hps_str, predictions: vector, fitted_attribues: dict) or
                              (hps_str, error: str)
   for transactions in the {prediction_month}
   where
     'fitted_attributes': dict of fitted attributes
         for en: 'coef_', 'interecept_'
         for gb: 'feature_importances_'
      A string represents that an exception occured. It is the text of the exception message.
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
    parser.add_argument('training_data', choices=['all', 'train'])
    parser.add_argument('neighborhood', type=neighborhood_type)
    parser.add_argument('model', choices=['en', 'gb', 'rf'])
    parser.add_argument('prediction_month', type=month_type)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--dry', action='store_true')     # don't write output
    arg = parser.parse_args(argv)
    arg.me = arg.invocation.split('.')[0] + '-v2'

    if arg.trace:
        pdb.set_trace()

    random_seed = 123
    random.seed(random_seed)

    dir_working = Path().dir_working()
    fit_dir = (
        os.path.join(dir_working, arg.me + '-test') if arg.test else
        os.path.join(dir_working, arg.me)
    )
    result_dir = '%s-%s-%s-%s' % (arg.training_data, arg.neighborhood, arg.model, arg.prediction_month)
    path_out_dir = os.path.join(fit_dir, result_dir, '')
    dirutility.assure_exists(path_out_dir)

    return Bunch(
        arg=arg,
        path_in_query_samples=os.path.join(dir_working, 'samples2', 'all.csv'),
        path_in_training_samples=os.path.join(dir_working, 'samples2', arg.training_data + '.csv'),
        path_out_actuals=os.path.join(path_out_dir, 'actuals.pickle'),
        path_out_transaction_ids=os.path.join(path_out_dir, 'transaction_ids.pickle'),
        path_out_predictions_attributes=os.path.join(path_out_dir, "predictions-attributes.pickle"),
        path_out_dir=path_out_dir,
        path_out_feature_names=os.path.join(path_out_dir, 'feature_names.pickle'),
        path_out_log=os.path.join(path_out_dir, '0log.txt'),
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
    'write fitted models to file system'
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

    def read_csv(path):
        df = pd.read_csv(
            path,
            nrows=100 if control.arg.test else None,
            usecols=None,  # TODO: change to columns we actually use
            low_memory=False
        )
        print 'read %d samples from file %s' % (len(df), path)
        return df

    def in_prediction_month(query_samples, prediction_YYYYMM):
        'return DataFrame of sample in the month we are predicting'
        def splitYYYYMMDD(dates):
            year_factor = 10000.0
            years = (dates / year_factor).astype('int64')
            month_factor = 100.0
            months = ((dates - years * year_factor) / month_factor).astype('int64')
            return years, months

        def splitYYYYMM(date_str):
            date = int(date_str)
            year_factor = 100.0
            year = int(date / year_factor)
            month = int(date - year * year_factor)
            return year, month

        sale_dates = query_samples[layout_transactions.sale_date]
        query_years, query_months = splitYYYYMMDD(sale_dates)
        prediction_year, prediction_month = splitYYYYMM(prediction_YYYYMM)
        mask_year = query_years == prediction_year
        mask_month = query_months == prediction_month
        mask = mask_year & mask_month
        result = query_samples.loc[mask]
        return result

    # reduce process priority, to try to keep the system responsive
    lower_priority()

    with open(control.path_out_feature_names, 'w') as f:
        feature_names = Features().ege_names('swpn')
        pickle.dump(feature_names, f)

    training_samples = read_csv(control.path_in_training_samples)

    query_samples_all = read_csv(control.path_in_query_samples)
    query_samples = in_prediction_month(query_samples_all, control.arg.prediction_month)
    print 'read %s query samples of which %d are in the prediction month %s' % (
        len(query_samples_all),
        len(query_samples),
        control.arg.prediction_month,
    )
    with open(control.path_out_transaction_ids, 'w') as f:
        transaction_ids = make_transaction_ids(query_samples)
        pickle.dump(transaction_ids, f)
    with open(control.path_out_actuals, 'w') as f:
        X, actuals = Features().extract_and_transform(query_samples, 'natural', 'natural')
        pickle.dump(actuals, f)

    count_fitted = 0
    n_hps = make_n_hps(control.arg.model)

    # determine hps we have already fitted and predicted
    already_seen = set()
    if os.path.exists(control.path_out_predictions_attributes):
        with open(control.path_out_predictions_attributes, 'r') as f:
            unpickler = pickle.Unpickler(f)
            try:
                while True:
                    hps_str, predictions, fitted_attributes = unpickler.load()
                    print 'existing', hps_str
                    already_seen.add(hps_str)
            except EOFError as e:
                pass
    print 'have already seen %d hps_str values' % len(already_seen)

    # fit and predict HPs that we have not already seen
    with open(control.path_out_predictions_attributes, 'w') as results_file:
        pickler = pickle.Pickler(results_file)
        for hps in HPs.iter_hps_model(control.arg.model):
            count_fitted += 1
            start_time = time.clock()  # wall clock time on Windows, processor time on Unix
            hps_str = HPs.to_str(hps)
            if hps_str in already_seen:
                print 'skipping already seen: %s' % hps_str
                continue
            try:
                predictions, fitted_attributes, n_training_samples = fit_and_predict(
                    training_samples,
                    query_samples,
                    hps, control,
                )
                pickler.dump((hps_str, predictions, fitted_attributes))
                pickler.clear_memo()  # don't build up a large data structure
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
                pickler.dump((hps_str, e))

            # collect to get memory usage stable, so that we can run this program many time in parallel
            gc.collect()
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
