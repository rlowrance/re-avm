'''analyze WORKING/samples2/train.csv (the training data) to select cities used to build local models

This program displays statistics on all cities in the training data. The investigator decides
which cities to select based on these statistics. These choices are recorded in the file
WORKING/select-cities/selected-cities.csv

reduce all the fit-predict output into a single large CSV file with all predictions

INVOCATION
  python select-cities.py [--test] [--trace]

INPUTS
 WORKING/samples2/train.csv

OUTPUTS
 WORKING/select-cities/report_by_n_trades.txt
 WORKING/select-cities/report_by_prices.txt
 WORKING/select-cities/0log.txt
'''

from __future__ import division

import argparse
import collections
import numpy as np
import os
import pandas as pd
import pdb
from pprint import pprint
import random
import sys

from Bunch import Bunch
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
    parser.add_argument('invocation')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')
    arg = parser.parse_args(argv)
    arg.me = arg.invocation.split('.')[0]

    if arg.trace:
        pdb.set_trace()

    random_seed = 123
    random.seed(random_seed)

    dir_working = Path().dir_working()
    dir_out = os.path.join(dir_working, arg.me + ('-test' if arg.test else ''))
    dirutility.assure_exists(dir_out)

    return Bunch(
        arg=arg,
        path_in_samples=os.path.join(dir_working, 'samples2', 'train.csv'),
        # path_out_csv=os.path.join(dir_out, 'reduction.csv'),
        path_out_report_by_price=os.path.join(dir_out, 'report-by-price.txt'),
        path_out_report_by_n_trades=os.path.join(dir_out, 'report-by-n-trades.txt'),
        path_out_log=os.path.join(dir_out, '0log.txt'),
        random_seed=random_seed,
        timer=Timer(),
    )


def etl(path_in, nrows):
    '''return (median_price OrderedDict[city] float, n_trades OrderedDict[city] float)'''
    city_column = layout_transactions.city
    price_column = layout_transactions.price

    extracted = pd.read_csv(
        path_in,
        nrows=nrows,
        usecols=[city_column, price_column],
        low_memory=False
    )

    print 'read %d samples from file %s' % (len(extracted), path_in)

    # build trades by city
    median_price = {}
    n_trades = {}

    for city in set(extracted[city_column]):
        mask = extracted[city_column] == city
        in_city = extracted.loc[mask]
        assert len(in_city) > 0, city
        median_price[city] = in_city[price_column].median()
        n_trades[city] = len(in_city)
    return (
        collections.OrderedDict(sorted(median_price.items(), key=lambda x: x[1])),
        collections.OrderedDict(sorted(n_trades.items(), key=lambda x: x[1])),
    )


def make_reports(median_prices, n_trades, median_prices_indices, n_trades_indices):
    'return (report sorted by price, report sorted by n_trades)'
    def make_report(title, ordered_cities):
        def make_detail_line(city):
            return {
                'city': city,
                'median_price': median_prices[city],
                'median_price_index': median_prices_indices[city],
                'n_trades': n_trades[city],
                'n_trades_index': n_trades_indices[city],
            }

        c = ColumnsTable(
            (
                (
                    'city', 30, '%30s',
                    ('', '', '', '', '', 'City'),
                    'city name'
                ),
                (
                    'median_price', 7, '%7.0f',
                    ('', '', '', '', 'median', 'price'),
                    'median price in city'
                ),
                (
                    'median_price_index', 7, '%7.2f',
                    ('median', 'price', '/', 'overall', 'median', 'price'),
                    'median price as fraction of overall median price'
                ),
                (
                    'n_trades', 7, '%7.0f',
                    ('', '', '', '', 'number', 'trades'),
                    'number of trades across all months'
                ),
                (
                    'n_trades_index', 7, '%7.2f',
                    ('number', 'trades', '/ ', 'overall', 'median', 'trades'),
                    'median number trades as fraction of overall median number of trades'
                ),
            )
        )
        for city in ordered_cities:
            c.append_detail(**make_detail_line(city))
        c.append_legend(40)

        r = Report()
        r.append(title)
        r.append(' ')
        for line in c.iterlines():
            r.append(line)
        return r

    by_price = make_report(
        'Prices and Number of Trades Ordered by Median Price in City',
        median_prices.keys(),
    )
    by_n_trades = make_report(
        'Prices and Number of Trades Ordered by Number of Trades in City',
        n_trades.keys(),
    )

    return by_price, by_n_trades


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

    median_prices, n_trades = etl(control.path_in_samples, 100 if control.arg.test else None)

    # build versions using index numbers relative to median values
    median_prices_indices, median_price = make_indices(median_prices)
    print 'median_price', median_price
    n_trades_indices, median_n_trades = make_indices(n_trades)
    print 'median_n_trades', median_n_trades
    report_by_price, report_by_n_trades = make_reports(
        median_prices,
        n_trades,
        median_prices_indices,
        n_trades_indices,
    )
    pdb.set_trace()
    report_by_n_trades.write(control.path_out_report_by_n_trades)
    report_by_price.write(control.path_out_report_by_price)


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
