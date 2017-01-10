'''analyze WORKING/samples-train.csv

INVOCATION: python samples-train-analysis.py ARGS

INPUT FILES:
 WORKING/samples-train.csv

OUTPUT FILES:
 WORKING/ME/0log.txt         log file containing what is printed
 WORKING/ME/transactions.csv with columns apn | date | sequence | actual_price
'''

import argparse
import collections
import math
import multiprocessing
import numpy as np
import os
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
    path_out_dir = dirutility.assure_exists(dir_working + arg.me + ('-test' if arg.test else '') + '/')
    return Bunch.Bunch(
        arg=arg,
        path_in_samples=dir_working + 'samples-train.csv',
        path_out_log=path_out_dir + '0log.txt',
        path_out_csv=path_out_dir + 'transactions.csv',
        random_seed=random_seed,
        test=arg.test,
        timer=Timer.Timer(),
        )


def make_index(apn, date, sequence_number):
    return '%d-%d-%d' % (apn, date, sequence_number)


APN_Date = collections.namedtuple('APN_Date', 'apn date')
ColumnName = collections.namedtuple('ColumnName', 'apn date actual_price')


def column_names():
    'return names of columns in the input csv'
    return ColumnName(
        apn='APN UNFORMATTED_deed',
        date='SALE DATE_deed',
        actual_price='SALE AMOUNT_deed',
        )


def create_new_row(apn, date, sequence_number, row, column):
    'return DataFrame with one row'
    date = int(date)
    date_year = int(date / 10000)
    date_month = int((date - date_year * 10000) / 100)
    date_day = int(date - date_year * 10000 - date_month * 100)
    assert date == date_year * 10000 + date_month * 100 + date_day, date
    new_df = pd.DataFrame(
        data={
            'apn': int(apn),
            'date': date,
            'year': date_year,
            'month': date_month,
            'day': date_day,
            'sequence_number': sequence_number,
            'actual_price': row[column.actual_price],
        },
        index=[make_index(apn, date, sequence_number)],
    )
    return new_df


class DuplicateFinder(object):
    def __init__(self, df, column):
        self.df = df
        self.column = column

    def find_duplicates_method(self, apn_date):
        mask_apn = self.df[self.column.apn] == apn_date.apn
        mask_date = self.df[self.column.date] == apn_date.date
        mask = mask_apn & mask_date
        df_apn_date = self.df[mask]
        sequence_number = 0
        result_duplicates = set()
        result_df = None
        for i, row in df_apn_date.iterrows():
            if sequence_number > 0:
                result_duplicates.add(apn_date)
            new_df = create_new_row(apn_date.apn, apn_date.date, sequence_number, row, self.column)
            result_df = new_df if result_df is None else result_df.append(new_df, verify_integrity=True)
            sequence_number += 1
        return result_df, result_duplicates


Info = collections.namedtuple('Info', 'apn_date, df, column')
MappedItem = collections.namedtuple('MappedItem', 'transactions duplicates')


def make_transactions_mapper(info):
    'return (df of transactions, set of duplicates)'
    verbose = False
    if verbose:
        print 'make_transactions_mapper begin', info.apn_date, 'pid', os.getpid()
    apn_date, df, column = info.apn_date, info.df, info.column
    mask_apn = df[column.apn] == apn_date.apn
    mask_date = df[column.date] == apn_date.date
    mask = mask_apn & mask_date
    df_apn_date = df[mask]
    sequence_number = 0
    result_duplicates = set()
    result_df = pd.DataFrame()
    for label, row in df_apn_date.iterrows():
        if verbose:
            print 'make_transactions_mapper iter row label', info.apn_date, label
        if sequence_number > 0:
            result_duplicates.add(apn_date)
        new_df = create_new_row(apn_date.apn, apn_date.date, sequence_number, row, column)
        result_df = result_df.append(new_df, ignore_index=True, verify_integrity=True)
        # result_df = new_df if result_df is None else result_df.append(new_df, verify_integrity=True)
        sequence_number += 1
    if verbose:
        print 'make_transactions_mapper end', len(result_df), len(result_duplicates)
    return MappedItem(
        transactions=result_df,
        duplicates=result_duplicates,
        )


def make_transactions_reducer(mapped_items, sofar_transactions=pd.DataFrame(), sofar_duplicates=set()):
    'reduce list of results from mapper to final result'
    for mapped_item in mapped_items:
        transactions, duplicates = mapped_item.transactions, mapped_item.duplicates
        sofar_transactions = sofar_transactions.append(transactions, verify_integrity=True, ignore_index=True)
        if len(duplicates) > 0:
            sofar_duplicates = sofar_duplicates.update(duplicates)
            # otherwise set().update(set()) --> None (not set(), which would be reasonable)
    return sofar_transactions, sofar_duplicates


def make_transactions_parallel(df, test, in_parallel):
    'return (df of transaction with unique APNs and IDs, set(APN_Date) of duplicates)'
    # use 4 cores
    def make_infos():
        'return list of Info'
        column = column_names()
        result = []
        for apn in set(df[column.apn]):
            df_apn = df[df[column.apn] == apn]
            for date in set(df_apn[column.date]):
                result.append(Info(
                    apn_date=(APN_Date(apn, date)),
                    df=df,
                    column=column,
                ))
        return result

    all_infos = make_infos()
    print 'len(all_infos)', len(all_infos)
    infos = all_infos[:100] if test else all_infos
    print 'len(infos)', len(infos)

    if in_parallel:
        n_processes = 1
        print 'n_processes', n_processes
        p = multiprocessing.Pool(n_processes)
        mapped = p.map(make_transactions_mapper, infos)
        all_transactions, all_duplicates = make_transactions_reducer(
            mapped,
            sofar_transactions=pd.DataFrame(),
            sofar_duplicates=set(),
            )
    else:
        # conserve memory by processing each info one by one
        verbose = False
        all_transactions = pd.DataFrame()
        all_duplicates = set()
        for info in infos:
            if verbose:
                print all_transactions
                print all_duplicates
            mapped_item = make_transactions_mapper(info)
            if verbose:
                print mapped_item.transactions
                print mapped_item.duplicates
                print len(mapped_item)
            all_transactions, all_duplicates = make_transactions_reducer(
                [mapped_item],
                sofar_transactions=all_transactions,
                sofar_duplicates=all_duplicates,
                )
            assert len(all_transactions) > 0
            assert all_duplicates is not None
    return all_transactions, all_duplicates


def make_transactionsOLD(df, test):
    'return (df of transaction IDs and prices, set(APN_Date) of duplicates)'
    column = column_names()
    result_df = None
    result_set = set()
    for apn in set(df[column.apn]):
        if test and len(result_set) > 10:
            break
        df_apn = df[df[column.apn] == apn]
        for date in set(df_apn[column.date]):
            duplicates_df, duplicates_set = make_transactions_mapper(APN_Date(apn, date), df, column)
            result_df = duplicates_df if result_df is None else result_df.append(duplicates_df, verify_integrity=True, ignore_index=True)
            result_set.append(duplicates_set)
    return result_df, result_set


def make_how_different(df, duplicates):
    'return tuple (dict[column] = set((value0, value1)), matched_counter) of mismatched fields'
    def isnan(x):
        if isinstance(x, float):
            return math.isnan(x)
        if isinstance(x, np.float64):
            return np.isnan(x)
        return False

    def find_mismatched_values(ordered_columns, matched):
        'return None or (column, value0, value1) of mistmatched fields in first 2 records'
        # TODO: can compare all records until a mismatch is found
        match0 = matches.iloc[0]
        match1 = matches.iloc[1]
        for column in ordered_columns:
            value0 = match0[column]
            value1 = match1[column]
            # print value0, value1, type(value0), type(value1)
            if isnan(value0) and isnan(value1):
                # NaN stands for Missing in pandas.DataFrame
                continue  # pretend that two NaN values are equal to each other
            if value0 != value1:
                print column, value0, value1
                return column, value0, value1
        return None  # should not happen

    def make_ordered_columns(column, df):
        'return list of column names to examine'
        all_but_price = [
            column_name
            for column_name in df.columns
            if column_name not in (
                    column.actual_price,
                    'Unnamed: 0',
                    'Unnamed: 0.1',
            )
        ]
        ordered_columns = [column.actual_price]
        ordered_columns.extend(all_but_price)
        return ordered_columns

    column = column_names()
    ordered_columns = make_ordered_columns(column, df)
    result = collections.defaultdict(list)
    matched_counter = collections.Counter()
    for duplicate in duplicates:
        mask_apn = df[column.apn] == duplicate.apn
        mask_date = df[column.date] == duplicate.date
        mask = mask_apn & mask_date
        matches = df[mask]
        matched_counter[len(matches)] += 1
        if len(matches) > 1:
            maybe_mismatched_values = find_mismatched_values(ordered_columns, matches)
            if maybe_mismatched_values is None:
                print ' all fields in first 2 records were equal'
                pdb.set_trace()
            else:
                column_name, value0, value1 = maybe_mismatched_values
                result[column_name].append((value0, value1))
        else:
            print matches
            print duplicate
            print len(matches)
            print 'no mismatched fields'
            pdb.set_trace()
    return result, matched_counter


def do_work(control):
    df = pd.read_csv(
        control.path_in_samples,
        low_memory=False,
        nrows=1000 if control.test and False else None,
        )

    print 'column names in input file', control.path_in_samples
    for i, column_name in enumerate(df.columns):
        print i, column_name

    control.timer.lap('printed column names')
    transactions_df2, duplicates2 = make_transactions_parallel(df, control.test, False)
    print 'not parallel lens', len(transactions_df2), len(duplicates2)
    control.timer.lap('make transactions without parallel')
    transactions_df, duplicates = make_transactions_parallel(df, control.test, True)
    print 'parallel lens', len(transactions_df), len(duplicates)
    control.timer.lap('make  transactions in parallel')
    print 'number of duplicate apn|date values', len(duplicates)
    print 'number of training samples', len(df)
    print 'number of unique apn-date-sequence_numbers', len(transactions_df)
    transactions_df.to_csv(control.path_out_csv)

    how_different, matched_counter = make_how_different(df, duplicates)
    print 'first field difference in first 2 records of duplicate apn|date transactions'
    for column, values in how_different.iteritems():
        print column
        for value in values:
            print ' ', value
    print
    print 'number of matches in duplicate records'
    for num_matched, num_times in matched_counter.iteritems():
        print '%d records had identical APN and sale dates %d times' % (num_matched, num_times)
    return None


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
