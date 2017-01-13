'''predict using many fitted models on the real estate data

INVOCATION
  python fit.py DATA MODEL TRANSACTIONMONTH NEIGHBORHOOD
where
 SAMPLES           in {train, all} specifies which data in Working/samples2 to use
 MODEL             in {en, gb, rf} specified which model to use
 TRANSACTIONMONTH  like YYYYMM specfies the last month of training data
 NEIGHBORHOOD      in {all, city_name} specifies whether to train a model on all cities or just the specified city

which will fit all observations in the MONTH for all fitted models of kind MODEL

INPUTS
 WORKING/samples2/train.csv or WORKING/samples2/all.csv
 WORKING/fit/DATA-MODEL-{TRANSACTIONMONTH - 1}-NEIGHBHOOD/*.pickle  the fitted models

OUTPUTS
 WORKING/predict[-item]/SAMPLES-MODEL-TRANSACTIONMONTH-NEIGHBORHOOD/predictions.pickle  a dict
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
    parser.add_argument('samples', choices=['all', 'train'])
    parser.add_argument('model', choices=['en', 'gb', 'rf'])
    parser.add_argument('transaction_month')
    parser.add_argument('neighborhood')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')
    arg = parser.parse_args(argv)

    arg.me = arg.invocation.split('.')[0]

    if arg.trace:
        pdb.set_trace()

    # convert arg.neighborhood into arg.all and arg.city
    arg.city = (
        None if arg.neighborhood == 'all' else
        arg.neighborhood.replace('_', ' ')
    )

    random_seed = 123
    random.seed(random_seed)

    prior_month = Month(arg.transaction_month).decrement().as_str()
    in_dir = '%s-%s-%s-%s' % (arg.samples, arg.model, prior_month, arg.neighborhood)
    out_dir = '%s-%s-%s-%s' % (arg.samples, arg.model, arg.transaction_month, arg.neighborhood)

    dir_working = Path().dir_working()
    output_dir = (
        os.path.join(dir_working, arg.me + '-test', out_dir, '') if arg.test else
        os.path.join(dir_working, arg.me, out_dir, '')
    )
    dirutility.assure_exists(output_dir)

    return Bunch(
        arg=arg,
        path_in_fitted=os.path.join(dir_working, 'fit', in_dir, ''),
        path_in_samples=os.path.join(dir_working, 'samples2', arg.samples + '.csv'),
        path_out_file=os.path.join(output_dir, 'predictions.pickle'),
        path_out_log=os.path.join(output_dir, '0log.txt'),
        random_seed=random_seed,
        timer=Timer(),
    )


def do_work(control):
    'write predictions to output csv file'
    samples = pd.read_csv(
        control.path_in_samples,
        nrows=10 if control.arg.test else None,
        usecols=None,  # TODO: change to columns we actually use
        low_memory=False,
    )
    apns = samples[layout_transactions.apn]
    sale_dates = samples[layout_transactions.sale_date]
    print 'read %d rows of samples from file %s' % (len(samples), control.path_in_samples)

    # iterate over the fitted models
    hps_predictions = {}
    for root, dirnames, filenames in os.walk(control.path_in_fitted):
        assert len(dirnames) == 0, dirnames
        print root, len(filenames)
        for filename in filenames:
            suffix_we_process = '.pickle'
            if not filename.endswith(suffix_we_process):
                print 'skipping file without a fitted model: %s' % filename
                continue
            hps_string = filename[:-len(suffix_we_process)]
            hps = HPs.from_str(hps_string)
            path_to_file = os.path.join(root, filename)
            with open(path_to_file, 'r') as f:
                ok, fitted_model = pickle.load(f)
            if ok:
                print 'predicting samples using fitted model %s' % filename
                X, y = Features().extract_and_transform(samples, hps['units_X'], hps['units_y'])
                predictions = fitted_model.predict(X)
                assert len(predictions) == len(samples)
                assert hps_string not in hps_predictions
                hps_predictions[hps_string] = predictions
            else:
                print 'not not predict samples using fitted model %s; reason: %s' % (
                    filename,
                    fitted_model,  # an error message
                )
        # have all the predictions for all filenames (= a set of hyperparameters)
        print 'walked all %d files' % len(filenames)
    out = {
        'apns': apns,
        'sale_dates': sale_dates,
        'hps_predictions': hps_predictions,
    }
    with open(control.path_out_file, 'w') as f:
        pickle.dump(out, f)
    print 'wr0te results to %s' % control.path_out_file
    return


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
