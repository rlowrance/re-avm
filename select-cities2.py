'''analyze WORKING/samples2/train.csv (the training data) to select cities used to build local models

Summarize samples2/train.csv, select 12 cities
and use the summary to create a table of cities ordered by number of trades

INVOCATION
  python select-cities2.py n_cities [--test] [--trace]

INPUTS
 WORKING/samples2/train.csv

OUTPUTS
 WORKING/select-cities2/city-medianprice-ntrades.csv
 WORKING/select-cities2/city-medianprice-ntrades-all.txt
 WORKING/select-cities2/city-medianprice-ntrades-selected.txt
 WORKING/select-cities2/0log.txt
'''

from __future__ import division

import argparse
import collections
import json
import numpy as np
import os
import pandas as pd
import pdb
from pprint import pprint
import random
import sys

import arg_type
from Bunch import Bunch
import columns_table
from ColumnsTable import ColumnsTable
import dirutility
import layout_transactions
from Logger import Logger
from Path import Path
from Report import Report
from Timer import Timer


def make_control(argv):
    'return a Bunch'

    print argv
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')
    arg = parser.parse_args(argv)
    arg.me = parser.prog.split('.')[0]

    if arg.trace:
        pdb.set_trace()

    random_seed = 123
    random.seed(random_seed)
    np.random.seed(random_seed)

    dir_working = Path().dir_working()
    dir_out = os.path.join(dir_working, arg.me + ('-test' if arg.test else ''))
    dirutility.assure_exists(dir_out)

    base = 'city_medianprice_ntrades'
    base_all = base + '_all'
    base_selected = base + '_selected'

    return Bunch(
        arg=arg,
        path_in_column_defs=os.path.join('column_defs.json'),
        path_in_samples=os.path.join(dir_working, 'samples2', 'train.csv'),
        path_out_csv_all=os.path.join(dir_out, base_all + '.csv'),
        path_out_csv_selected=os.path.join(dir_out, base_selected + '.csv'),
        path_out_report_all=os.path.join(dir_out, base_all + '.txt'),
        path_out_report_selected=os.path.join(dir_out, base_selected + '.txt'),
        path_out_log=os.path.join(dir_out, '0log.txt'),
        random_seed=random_seed,
        timer=Timer(),
    )


def etl(path_in, nrows, test):
    'return DataFrames with columns city, median price, n trades: all and those selected'
    '''return (median_price OrderedDict[city] float, n_cities OrderedDict[city] float)'''
    city_column = layout_transactions.city
    price_column = layout_transactions.price

    extracted = pd.read_csv(
        path_in,
        nrows=nrows,
        usecols=[city_column, price_column],
        low_memory=False
    )

    print 'read %d samples from file %s' % (len(extracted), path_in)

    # build columns for the DataFrame result
    distinct_cities = set(extracted[city_column])
    selected_n_trades = (
        277, 296, 303, 351,        # about half the median
        638, 640, 642, 660,        # about the median number of trades (median is 641)
        4480, 5613, 10610, 22303,  # largest number of trades
    )

    cities = []
    median_prices = np.empty(len(distinct_cities))
    n_trades = np.empty(len(distinct_cities))
    selecteds = []

    for i, city in enumerate(distinct_cities):
        mask = extracted[city_column] == city
        in_city = extracted.loc[mask]
        assert len(in_city) > 0, city
        cities.append(city)
        median_prices[i] = in_city.median()
        n_trades[i] = len(in_city)
        selecteds.append(True if test else len(in_city) in selected_n_trades)

    # check that the counting by city is reasonable
    print 'sorted(n_trades)'
    print sorted(n_trades)
    print 'median', np.median(n_trades)
    if not test:
        assert sum(n_trades) == len(extracted)
        for selected_n_trade in selected_n_trades:
            assert selected_n_trade in n_trades, selected_n_trade

    result_all = pd.DataFrame(
        data={
            'city': cities,
            'median_price': median_prices,
            'n_trades': n_trades,
            'selected': selecteds,
        },
        index=cities,
    )
    result_selected = result_all.loc[result_all.selected]

    result_all_sorted = result_all.sort_values('n_trades')
    result_selected_sorted = result_selected.sort_values('n_trades')
    print result_selected_sorted
    return result_all_sorted, result_selected_sorted


def do_work(control):
    'create csv file that summarizes all actual and predicted prices'
    def make_indices(ordered_dict):
        'return OrderedDict[key] <index relative to median value of ordered_dict>'
        values = np.empty(len(ordered_dict), dtype=float)
        for i, value in enumerate(ordered_dict.values()):
            values[i] = value
        median_value = np.median(values)
        result = collections.OrderedDict()
        for k, v in ordered_dict.iteritems():
            result[k] = v / median_value
        return result, median_value

    df_all, df_selected = etl(
        control.path_in_samples,
        10 if control.arg.test else None,
        control.arg.test,
    )
    df_all.to_csv(control.path_out_csv_all)
    df_selected.to_csv(control.path_out_csv_selected)

    with open(control.path_in_column_defs, 'r') as f:
        column_defs = json.load(f)
        pprint(column_defs)

    def make_generate_data(df):
        'yield each input deail line as a dict-like object'
        def generate_data():
            for i, row in df.iterrows():
                yield row
        return generate_data

    def create_and_write(df, path, header_lines, selected_columns):
        'create report and write it'
        lines = columns_table.columns_table(
            make_generate_data(df)(),
            selected_columns,
            column_defs,
            header_lines,
        )
        with open(path, 'w') as f:
            for line in lines:
                f.write(line)

    create_and_write(
        df_all,
        control.path_out_report_all,
        ['Count of Trades in All Cities', 'Ordered by Count of Number of Trades'],
        ['city', 'median_price', 'n_trades', 'selected'],
    )
    create_and_write(
        df_selected,
        control.path_out_report_selected,
        ['Count of Trades in Selected Cities', 'Ordered by Count of Number of Trades'],
        ['city', 'median_price', 'n_trades'],
    )


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

    main(sys.argv[1:])
