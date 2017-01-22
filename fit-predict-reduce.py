'''reduce all the fit-predict output into a single large CSV file with all predictions

INVOCATION
  python fit-predict-reduce.py trainin_data n_processes [--test] [--trace] [--cache] [--testmapper]

where
 n_processes is the number of CPUs to alloccate
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
import collections
import cPickle as pickle
import gc
import multiprocessing
import os
import pandas as pd
import pdb
from pprint import pprint
import random
import sys
import time

import arg_type
from Bunch import Bunch
from Cache import Cache
import dirutility
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

    print argv
    parser = argparse.ArgumentParser()
    parser.add_argument('invocation')
    parser.add_argument('training_data', choices=arg_type.training_data_choices())
    parser.add_argument('neighborhood', type=arg_type.neigborhood)
    parser.add_argument('model', choices=arg_type.model_choices)
    parser.add_argument('n_processes', type=arg_type.n_processes)
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--testmapper', action='store_true')
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
        path_out_dir=dir_out,
        path_out_fitted_attributes=os.path.join(dir_out, 'fitted-attributes.pickle'),
        path_out_log=os.path.join(dir_out, '0log.txt'),
        random_seed=random_seed,
        timer=Timer(),
    )


MapperArg = collections.namedtuple(
    'MapperArg',
    'in_dir out_path_actuals_predictions out_path_fitted_attributes fitted_dirname, test',
    )
MapperResult = collections.namedtuple(
    'MapperResult',
    'mapper_arg ok n_rows_written',
    )


def mapper(mapper_arg):
    'return (mapper_arg, n_rows) and write a CSV file to mapper_arg.out_path'
    def load_pickled(dir, filename_base):
        path = os.path.join(dir, filename_base + '.pickle')
        with open(path, 'r') as f:
            result = pickle.load(f)
        return result

    def file_is_readable(path):
        'return True or False'
        if os.path.isfile(path):
            try:
                f = open(path, 'r')
                f.close()
                return True
            except:
                return False
        print 'path is not a file', path
        pdb.set_trace()
        return False

    def make_rows(actuals, dirname, hps_str, predictions, transaction_ids):
        'for now, return pd.DataFrame; later return several DataFrames'
        def in_prediction_month(sale_date, prediction_month):
            factor_year = 10000.0
            factor_month = 100.0
            sale_date_year = int(sale_date / factor_year)
            sale_date_month = int((sale_date - sale_date_year * factor_year) / factor_month)
            return sale_date_year == prediction_month.year and sale_date_month == prediction_month.month

        debug = False
        assert len(actuals) == len(predictions)
        assert len(actuals) == len(transaction_ids)
        print 'making %d rows dirname %s hps %s' % (len(actuals), dirname, hps_str)
        training_data, neighborhood, model, prediction_month_str = dirname.split('-')
        prediction_month = Month(prediction_month_str)
        hps = HPs.from_str(hps_str)
        result = pd.DataFrame()
        for i in xrange(len(actuals)):
            sale_date = transaction_ids[i].sale_date
            assert in_prediction_month(sale_date, prediction_month), (sale_date, prediction_month)
            row = {
                # transaction ID
                'transaction_id': transaction_ids[i],
                'apn': transaction_ids[i].apn,
                'sale_date': transaction_ids[i].sale_date,
                # what model was trained
                'dirname': dirname,
                'training_data': training_data,
                'neighborhood': neighborhood,
                'model': model,
                'prediction_month': prediction_month_str,
                # hyperparameters for that model
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
            if debug and len(result) > 10:
                print 'truncated rows, since debugging'
                break
        return result

    # BODY STARTS HERE
    debug = False
    start_time = time.time()
    print 'mapper', mapper_arg

    lower_priority()  # try to make the system usable for interactive work

    # read all of the short files
    actuals = load_pickled(mapper_arg.in_dir, 'actuals')
    feature_names = load_pickled(mapper_arg.in_dir, 'feature_names')
    transaction_ids = load_pickled(mapper_arg.in_dir, 'transaction_ids')
    print 'for now, skipped %d feature names' % len(feature_names)

    # read the long file record by record
    path = os.path.join(mapper_arg.in_dir, 'predictions-attributes.pickle')
    if not file_is_readable(path):
        return MapperResult(
            ok=False,
            error='skipping file that is not openable: %s' % path,
        )

    records_processed = 0
    result = pd.DataFrame()
    all_fitted_attributes = {}
    with open(path, 'r') as f:
        unpickler = pickle.Unpickler(f)
        try:
            while True:
                obj = unpickler.load()
                assert len(obj) in (2, 3), (obj, len(obj))
                if len(obj) == 2:
                    hps_str, error_message = obj
                    print 'skipping %s: error %' % (hps_str, error_message)
                else:
                    hps_str, predictions, fitted_attributes = obj
                    all_fitted_attributes[(mapper_arg.fitted_dirname, hps_str)] = fitted_attributes
                    rows = make_rows(
                        actuals,
                        mapper_arg.fitted_dirname,
                        hps_str,
                        predictions,
                        transaction_ids,
                    )
                    # print 'added %d rows from %s' % (len(rows), path)
                    result = result.append(rows, ignore_index=True)
                    records_processed += 1
                    gc.collect()
                if debug and records_processed >= 2:
                    print 'breaking because we are debugging'
                    break
        except EOFError as e:
            print 'EOFError raised after %d records processes: %s' % (records_processed, e)
    print 'created %d result records in %f wallclock seconds for %s' % (
        len(result),
        time.time() - start_time,
        mapper_arg,
    )
    # write files
    result.to_csv(mapper_arg.out_path_actuals_predictions)
    with open(mapper_arg.out_path_fitted_attributes, 'w') as f:
        pickle.dump(all_fitted_attributes, f)
    return MapperResult(
        mapper_arg=mapper_arg,
        ok=True,
        n_rows_written=len(result),
    )


def do_work(control):
    'create csv file that summarizes all actual and predicted prices'
    def read_csv(path):
        df = pd.read_csv(
            path,
            nrows=8000 if control.arg.test else None,
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
        all = transaction_id_set(control.path_in_query_samples_all)
        train = transaction_id_set(control.path_in_query_samples_train)
        pdb.set_trace()
        query_in_testing_set = set()
        query_in_training_set = set()
        for query in all:
            if query in train:
                query_in_training_set.add(query)
            else:
                query_in_testing_set.add(query)
        return query_in_testing_set, query_in_training_set

    def dirname_training_data(dirname):
        training_data, neighbhood, model, prediction_month = dirname.split('-')
        return training_data

    def dirname_neighborhood(dirname):
        training_data, neighbhood, model, prediction_month = dirname.split('-')
        return nieghborhood

    def dirname_training_data(dirname):
        training_data, neighbhood, model, prediction_month = dirname.split('-')
        return model

    # BODY STARTS HERE
    # determine training and testing transactions
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

    # create DataFrame in subprocess for each input directory in dirnames below
    dirpath, dirnames, filenames = next(os.walk(control.path_in_dir_fit_predict))
    print 'will process %d dirnames in %d processes' % (len(dirnames), control.arg.n_processes)
    pool = multiprocessing.Pool(control.arg.n_processes)
    worker_args = [
        MapperArg(
            in_dir=os.path.join(dirpath, dirname),
            out_path_actuals_predictions=os.path.join(control.path_out_dir, dirname + '-actuals-predictions.csv'),
            out_path_fitted_attributes=os.path.join(control.path_out_dir, dirname + '-fitted-attributes.pickle'),
            fitted_dirname=dirname,
            test=control.arg.test
        )
        for dirname in dirnames
        if dirname_training_data(dirname) == control.arg.training_data
        if dirname_neighborhood(dirname) == control.arg.neighborhood
        if dirname_model(dirname) == control.arg.model
    ]
    print 'len(worker_args):', len(worker_args)
    pdb.set_trace()
    # mapped_results is a list of results from the mapper
    mapped_results = (
        [mapper(worker_args[0])] if control.arg.testmapper else
        pool.map(mapper, worker_args)
    )
    # reduce the mapped results, which are mainly in the file system
    print 'mapped_results'
    for mapped_result in mapped_results:
        # print mapped_result.mapper_arg
        if mapped_result.ok:
            print '%s succesfully created %d rows' % (
                mapped_result.mapper_arg.fitted_dirname,
                mapped_result.n_rows_written,
            )
        else:
            print 'bad result', mapped_result.error
    # TODO: combine the results, which are in files, into one big DataFrame
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

    main(sys.argv)
