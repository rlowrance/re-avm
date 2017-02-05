'''create samples that have no duplicate apn|date keys

INVOCATION: python samples2.py ARGS

INPUT FILES:
 WORKING/samples-train.csv
 WORKING/samples-test.csv

OUTPUT FILES:
 WORKING/samples2/0log.txt           log file containing what is printed
 WORKING/samples2/duplicates.pickle  duplicate TransactionId set
 WORKING/samples/uniques.pickle      unique TransactionId set
 WORKING/samples2/test.csv           enques transactions from samples-test.csv
 WORKING/samples2/train.csv          uniques transactions from samples-train.csv
 WORKING/samples2/all.csv            unique transactions from samples-test and sampes-train
 WORKING/samples2/actauls-all.csv    transaction_ids & actual prices for all transactions
 WORKING/samples2/actauls-test.csv   transaction_ids & actual prices for test transactions
 WORKING/samples2/actauls-train.csv  transaction_ids & actual prices for train transactions
'''

import argparse
import collections
import cPickle as pickle
import numpy as np
import os
import pandas as pd
import pdb
import random
import sys

import Bunch
import dirutility
import layout_transactions
import Logger
import Path
import Timer
import TransactionId


def make_control(argv):
    print 'argv', argv
    parser = argparse.ArgumentParser()
    parser.add_argument('invocation')
    parser.add_argument('--test', action='store_true', help='if present, truncated input and enable test code')
    parser.add_argument('--trace', action='store_true', help='if present, call pdb.set_trace() early in run')
    arg = parser.parse_args(argv)  # ignore invocation name
    arg.me = arg.invocation.split('.')[0]

    if arg.trace:
        pdb.set_trace()

    random_seed = 123
    random.seed(random_seed)

    dir_working = Path.Path().dir_working()
    path_out_dir = (
        os.path.join(dir_working, arg.me + '-test', '') if arg.test else
        os.path.join(dir_working, arg.me, '')
    )
    dirutility.assure_exists(path_out_dir)
    # path_out_dir = dirutility.assure_exists(dir_working + arg.me + ('-test' if arg.test else '') + '/')
    return Bunch.Bunch(
        arg=arg,
        path_in_test=os.path.join(dir_working, 'samples-test.csv'),
        path_in_train=os.path.join(dir_working, 'samples-train.csv'),
        path_out_log=os.path.join(path_out_dir, '0log.txt'),
        path_out_test=os.path.join(path_out_dir, 'test.csv'),
        path_out_train=os.path.join(path_out_dir, 'train.csv'),
        path_out_all=os.path.join(path_out_dir, 'all.csv'),
        path_out_duplicates=os.path.join(path_out_dir, 'duplicates.pickle'),
        path_out_uniques=os.path.join(path_out_dir, 'uniques.pickle'),
        random_seed=random_seed,
        test=arg.test,
        timer=Timer.Timer(),
        )


def read_extract_transform(path, nrows):
    'return (DataFrame with created transaction_id, collections.counter of transaction_ids)'
    df = pd.read_csv(path, low_memory=False, nrows=nrows)
    canonical = TransactionId.canonical
    TId = TransactionId.TransactionId
    id_count = collections.Counter()
    transaction_ids = []
    for index, row in df.iterrows():
        transaction_id = canonical(TId(
            apn=row[layout_transactions.apn],
            sale_date=row[layout_transactions.sale_date],
        ))
        transaction_ids.append(transaction_id)
        id_count[transaction_id] += 1
    df[layout_transactions.transaction_id] = transaction_ids
    df.index = transaction_ids
    return df, id_count


def make_uniques_dups(counts_a, counts_b):
    all_counts = collections.Counter()
    all_counts.update(counts_a)
    all_counts.update(counts_b)
    uniques = set()
    dups = set()
    for id in all_counts:
        if all_counts[id] == 1:
            assert id not in uniques
            uniques.add(id)
        else:
            assert id not in dups
            dups.add(id)
    return uniques, dups


def select_uniques(df, uniques):
    'return df containing only unique transactions; add transaction_id; set index to transaction_id'
    has_unique_transaction_id = [
        transaction_id in uniques
        for transaction_id in df['transaction_id']
        ]
    result = df.loc[has_unique_transaction_id]
    return result


def do_work(control):
    # read input files
    nrows = 2000 if control.test else None  # 2000 will find duplicates, 1000 will not
    in_test_df, in_test_counts = read_extract_transform(control.path_in_test, nrows)
    in_train_df, in_train_counts = read_extract_transform(control.path_in_train, nrows)
    in_all = in_train_df.append(in_test_df)

    # determine unique transaction ids

    uniques, duplicates = make_uniques_dups(in_test_counts, in_train_counts)

    # write unique transactions
    out_test_df = select_uniques(in_test_df, uniques)
    out_train_df = select_uniques(in_train_df, uniques)
    out_all_df = select_uniques(in_all, uniques)

    out_test_df.to_csv(control.path_out_test)
    out_train_df.to_csv(control.path_out_train)
    out_all_df.to_csv(control.path_out_all)

    # write unique and duplicate
    with open(control.path_out_duplicates, 'w') as f:
        pickle.dump(duplicates, f)
    with open(control.path_out_uniques, 'w') as f:
        pickle.dump(uniques, f)


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger.Logger(control.path_out_log)  # now print statements also write to the log file
    print control
    lap = control.timer.lap

    do_work(control)

    lap('work completed')
    if control.test:
        print 'DISCARD OUTPUT: test'
    print control
    print 'done'
    return


if __name__ == '__main__':
    main(sys.argv)
    if False:
        np
        pdb
        pd
