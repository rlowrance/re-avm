'''analyze WORKING/samples-train.csv

INVOCATION: python samles-train-analys.py ARGS

INPUT FILES:
 WORKING/samples-train.csv

OUTPUT FILES:
 WORKING/ME/0log.txt         log file containing what is printed
 WORKING/ME/transactions.csv with columns apn | date | sequence | actual_price
'''

import argparse
import collections
import numpy as np
import pandas as pd
import pdb
import random
import sys

import Bunch
import dirutility
import Logger
import Month
import Path
import Timer


def make_control(argv):
    print 'argv', argv
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='if present, truncated input and enable test code')
    parser.add_argument('--trace', action='store_true', help='if present, call pdb.set_trace() early in run')
    arg = parser.parse_args(argv[1:])  # ignore invocation name
    arg.me = 'samples-train-analysis'

    if arg.trace:
        pdb.set_trace()

    random_seed = 123
    random.seed(random_seed)

    dir_working = Path.Path().dir_working()
    path_out_dir = dirutility.assure_exists(dir_working + arg.me + ('-test/' if arg.test else '') + '/')
    return Bunch.Bunch(
        arg=arg,
        path_in_samples=dir_working + 'samples-train.csv',
        path_out_log=path_out_dir + '0log.txt',
        path_out_csv=path_out_dir + 'transactions.csv',
        random_seed=random_seed,
        validation_month=Month.Month(2005, 12),
        test=arg.test,
        timer=Timer.Timer(),
        )


def make_index(apn, date, sequence_number):
    return '%d-%d-%d' % (apn, date, sequence_number)


def do_work(control):
    df = pd.read_csv(
        control.path_in_samples,
        low_memory=False,
        nrows=1000 if control.test else None,
        )
    print 'column names in input file', control.path_in_samples
    for i, column_name in enumerate(df.columns):
        print i, column_name
    column_apn = 'APN UNFORMATTED_deed'
    column_date = 'SALE DATE_deed'
    column_actual_price = 'SALE AMOUNT_deed'
    result = None
    n_duplicates = 0
    for apn in set(df[column_apn]):
        df_apn = df[df[column_apn] == apn]
        for date in set(df_apn[column_date]):
            df_apn_date = df_apn[df_apn[column_date] == date]
            sequence_number = 0
            for i, row in df_apn_date.iterrows():
                if sequence_number > 0:
                    print 'duplicate apn|date', apn, date
                    n_duplicates += 1
                new_df = pd.DataFrame(
                    data={
                        'apn': int(apn),
                        'date': int(date),
                        'sequence_number': sequence_number,
                        'actual_price': row[column_actual_price],
                        },
                    index=[make_index(apn, date, sequence_number)],
                    )
                result = new_df if result is None else result.append(new_df, verify_integrity=True)
                sequence_number += 1
    print 'number of duplicate apn|date values', n_duplicates
    print 'number of training samples', len(df)
    print 'number of unique apn-date-sequence_numbers', len(result)
    result.to_csv(control.path_out_csv)


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger.Logger(logfile_path=control.path_out_log)  # now print statements also write to the log file
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
