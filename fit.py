'''fit many models on the real estate data

INVOCATION
  python fit.py DATA MODEL LASTMONTH NIEHGBORHOOD
where
 DATA         in {train, all} specifies which data in Working/samples2 to use
 MODEL        in {en, gb, rf} specified which model to use
 LASTMONTH    like YYYYMM specfies the last month of training data
 NEIGHBORHOOD in {all, city_name} specifies whether to train a model on all cities or just the specified city

INPUTS
 WORKING/samples2/train.csv or
 WORKING/samples2/all.csv

OUTPUTS
 WORKING/fit[-test]/DATA-MODEL-LASTMONTH-NEIGHBORHOOD/<filename>.pickle
where 
 <filename> is the hyperparameters for the model
 the pickle files contains either
  (True, <fitted-model>)
  (False, <error message explaining why model could not be fitted)
'''

from __future__ import division

import argparse
import collections
import cPickle as pickle
import datetime
import itertools
import numpy as np
import os
import pandas as pd
import pdb
from pprint import pprint
import random
import sklearn
import sys
import time

import arg_type
import AVM
from Bunch import Bunch
from columns_contain import columns_contain
import dirutility
from Features import Features
import HPs
import layout_transactions
from Logger import Logger
from Month import Month
from Path import Path
from Report import Report
from SampleSelector import SampleSelector
from valavmtypes import ResultKeyEn, ResultKeyGbr, ResultKeyRfr, ResultValue
from Timer import Timer
# from TimeSeriesCV import TimeSeriesCV
cc = columns_contain


def make_control(argv):
    'return a Bunch'

    print argv
    parser = argparse.ArgumentParser()
    parser.add_argument('invocation')
    parser.add_argument('data', choices=['all', 'train'])
    parser.add_argument('model', choices=['en', 'gb', 'rf'])
    parser.add_argument('last_month')
    parser.add_argument('neighborhood')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')
    arg = parser.parse_args(argv)
    arg.me = arg.invocation.split('.')[0]

    if arg.trace:
        pdb.set_trace()

    arg.last = Month(arg.last_month)  # convert to Month and validate value

    # convert arg.neighborhood into arg.all and arg.city
    arg.city = (
        None if arg.neighborhood == 'all' else
        arg.neighborhood.replace('_', ' ')
    )

    random_seed = 123
    random.seed(random_seed)

    dir_working = Path().dir_working()
    fit_dir = (
        os.path.join(dir_working, arg.me + '-test') if arg.test else
        os.path.join(dir_working, arg.me)
    )
    last_dir = '%s-%s-%s-%s' % (arg.data, arg.model, arg.last_month, arg.neighborhood)
    path_out_dir = os.path.join(fit_dir, last_dir, '')
    dirutility.assure_exists(path_out_dir)

    return Bunch(
        arg=arg,
        path_in_dir=os.path.join(dir_working, 'samples2', ''),
        path_out_dir=path_out_dir,
        path_out_log=os.path.join(path_out_dir, '0log.txt'),
        random_seed=random_seed,
        timer=Timer(),
    )


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
    cities = df[layout_transactions.city]
    mask = cities == city
    df_in_city = df.loc[mask]
    return df_in_city


def select_in_time_period_and_in_city(df, last_month, n_months_back, city):
    'return new df with the specified training data'
    verbose = False
    in_time_period = select_in_time_period(
        df.copy(),
        last_month,
        n_months_back,
    )
    in_neighborhood = (
        in_time_period if city is None else
        select_in_city(in_time_period, city)
    )
    if verbose:
        print 'neighborhood %s: %d in time period, %d also in neighborhood' % (
            'all' if city is None else city,
            len(in_time_period),
            len(in_neighborhood),
        )
    return in_neighborhood


def fit_en(dir_out, x, y, hps, random_seed, timer):
    'write one fitted model'
    verbose = False
    start_time = time.clock()
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
    fitted = model.fit(x, y)
    out_filename = HPs.to_str(hps)
    path_out = os.path.join(dir_out, out_filename + '.pickle')
    with open(path_out, 'w') as f:
        obj = (True, fitted)  # True ==> fitted successfully
        pickle.dump(obj, f)
    if verbose:
        print 'fit and write en %s in wallclock secs %s' % (
            out_filename,
            time.clock() - start_time,
        )


def do_work(control):
    'write fitted models to file system'
    path_in = os.path.join(control.path_in_dir, control.arg.data + '.csv')
    training_data = pd.read_csv(
        path_in,
        nrows=10 if control.arg.test else None,
        usecols=None,  # TODO: change to columns we actually use
        low_memory=False,
    )
    print 'read %d rows of training data from file %s' % (len(training_data), path_in)
    count_fitted = 0
    for hps in HPs.iter_hps_model(control.arg.model):
        start_time = time.clock()
        relevant = select_in_time_period_and_in_city(
            training_data,
            control.arg.last_month,
            hps['n_months_back'],
            control.arg.city,
        )
        if len(relevant) == 0:
            print 'skipping fitting of model, because no training samples for those hyperparameters'
            continue

        X, y = Features().extract_and_transform(
            relevant,
            hps['units_X'],
            hps['units_y'],
        )

        if control.arg.model == 'en':
            fit_en(control.path_out_dir, X, y, hps, control.random_seed, control.timer)
        elif control.arg.model == 'gb':
            pdb.set_trace()
            fit_gb(control.path_out_dir, X, y, hps, control.random_seed, control.timer)
        else:
            pdb.set_trace()
            fit_rf(control.path_out_dir, X, y, hps, control.random_seed, control.timer)
        count_fitted += 1
        print 'fitted #%4d on:%6d in: %6.2f %s %s %s %s hps: %s ' % (
            count_fitted,
            len(relevant),
            time.clock() - start_time,            control.arg.data,
            control.arg.model,
            control.arg.last_month,
            control.arg.neighborhood,
            HPs.to_str(hps),

        )
        if control.arg.test and count_fitted == 5:
            print 'breaking because we are tracing'
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
        pd.DataFrame()
        np.array()

    main(sys.argv)
