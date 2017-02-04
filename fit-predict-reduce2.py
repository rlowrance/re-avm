'''reduce all the fit-predict output. Convert transaction_ids to have datetime.dates

INVOCATION
  python fit-predict-reduce.py [--test] [--trace]

INPUTS
 WORKING/samples2/train.csv
 WORKING/samples2/all.csv
 WORKING/fit-predict-v2/{training_data}-{neighborhood}-{model}-{prediction_month}/transaction_ids.pickle
 WORKING/fit-predict-v2/{training_data}-{neighborhood}-{model}-{prediction_month}/prediction-attributes.pickle

OUTPUTS
 WORKING/fit-predict-reduce2/reduction.pickle: Dict[(date,apn), Dict[(fitted, hps_str), prediction]]
 WORKING/fit-predict-reduce2/reduction_2007.pickle: Dict[(date,apn), Dict[(fitted, hps_str), prediction]]
 WORKING/fit-predict-reduce2/no_data.pickle: Set[dirname]  # dirnames without any data (must be refitted)

OPERATIONAL NOTES:
 single threaded
'''

from __future__ import division

import argparse
import cPickle as pickle
import numpy as np
import os
import pdb
from pprint import pprint
import random
import sys

from Bunch import Bunch
import dirutility
from Fitted import Fitted
import HPs
from Logger import Logger
from Month import Month
from Path import Path
from Timer import Timer
import TransactionId


def make_control(argv):
    'return a Bunch'

    print argv
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')
    arg = parser.parse_args(argv[1:])
    arg.me = parser.prog.split('.')[0]

    if arg.trace:
        pdb.set_trace()

    random_seed = 123
    random.seed(random_seed)

    dir_working = Path().dir_working()
    dir_out = os.path.join(dir_working, arg.me)
    dirutility.assure_exists(dir_out)

    return Bunch(
        arg=arg,
        path_in_dir=os.path.join(dir_working, 'fit-predict-v2'),
        path_out_no_data=os.path.join(dir_out, 'no_data.pickle'),
        path_out_reduction=os.path.join(dir_out, 'reduction.pickle'),
        path_out_reduction_2007=os.path.join(dir_out, 'reduction_2007.pickle'),
        path_out_reduction_200701=os.path.join(dir_out, 'reduction_200701.pickle'),
        path_out_log=os.path.join(dir_out, '0log.txt'),
        random_seed=random_seed,
        timer=Timer(),
    )


def read_transaction_ids(dirpath, dirname):
    'return tuple of transaction ids'
    path = os.path.join(dirpath, dirname, 'transaction_ids.pickle')
    with open(path, 'r') as f:
        reduction = pickle.load(f)
    return tuple(reduction)


def process_dirname(dirpath, dirname, reduction, reduction_2007, reduction_200701, no_data, test):
    'mutate result and no_data to include info in the transactions and predictions files in dirname'
    verbose = False
    if verbose:
        print dirname
    training_data, neighborhood, model, month_str = dirname.split('-')
    if model == 'gb':
        print 'for now, skipping gb', dirname
        return
    month = Month(month_str)
    in_2007 = month.year == 2007
    in_200701 = month.year == 2007 and month.month == 1
    fitted = Fitted(training_data, neighborhood, model)
    transaction_ids_raw = read_transaction_ids(dirpath, dirname)
    transaction_ids_list = []
    for transaction_id_raw in transaction_ids_raw:
        canonical = TransactionId.canonical(transaction_id_raw)
        transaction_ids_list.append(canonical)
    transaction_ids = tuple(transaction_ids_list)
    path = os.path.join(dirpath, dirname, 'predictions-attributes.pickle')
    n_records_processed = 0
    with open(path, 'r') as f:
        unpickler = pickle.Unpickler(f)
        dirname_reduction = {}
        try:
            while True:
                obj = unpickler.load()
                if len(obj) == 3:
                    hps_str, predictions, fitted_attributes = obj
                    # convert from log domain to natural units
                    units_y = HPs.from_str(hps_str)['units_y']
                    predictions_restated = np.exp(predictions) if units_y == 'log' else predictions
                    if verbose:
                        print hps_str
                    dirname_reduction[hps_str] = predictions_restated
                else:
                    print 'error:', obj
                n_records_processed += 1
                if test and n_records_processed >= 10:
                    if verbose:
                        print 'test: stop after', n_records_processed
                    break
        except EOFError as e:
            if verbose:
                print 'EOFError (%s) for %s after %d records processed' % (e, dirname, n_records_processed)
            if n_records_processed == 0:
                no_data.add(dirname)
        except ValueError as e:
            print '%s' % e
            no_data.add(dirname)
    reduction_key = (fitted, transaction_ids)
    reduction[reduction_key] = dirname_reduction
    if in_2007:
        reduction_2007[reduction_key] = dirname_reduction
    if in_200701:
        reduction_200701[reduction_key] = dirname_reduction
    print dirname, n_records_processed, len(reduction), len(reduction_2007), len(no_data)


def do_work(control):
    reduction = {}
    reduction_2007 = {}
    reduction_200701 = {}
    no_data = set()
    for dirpath, dirnames, filenames in os.walk(control.path_in_dir):
        for dirname in dirnames:
            process_dirname(
                dirpath, dirname,
                reduction, reduction_2007, reduction_200701,
                no_data,
                control.arg.test,
            )
    print 'writing output files'
    with open(control.path_out_reduction, 'w') as f:
        pickle.dump(reduction, f)
    with open(control.path_out_reduction_2007, 'w') as f:
        pickle.dump(reduction_2007, f)
    with open(control.path_out_reduction_200701, 'w') as f:
        pickle.dump(reduction_200701, f)
    with open(control.path_out_no_data, 'w') as f:
        pickle.dump(no_data, f)
    print 'found %d dirnames without data (need to refit these models)' % len(no_data)
    for dirname in no_data:
        print dirname


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
