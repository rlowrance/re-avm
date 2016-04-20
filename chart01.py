'''create charts showing median and mean prices each month

INVOCATION
  python chart01.py [--data] [--test]

INPUT FILES
 INPUT/samples-train-validate.csv

OUTPUT FILES
 WORKING/chart01/data.pickle   # dict: keys=ReductionKey values=ReductionValue
 WORKING/chart01/median-price.pdf
 WORKING/chart01/median-price.txt
 WORKING/chart01/median-price_2006_2007.txt
'''

from __future__ import division

import argparse
import collections
import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pdb
from pprint import pprint
import random
import sys

from Bunch import Bunch
from columns_contain import columns_contain
from Logger import Logger
from Path import Path
from Report import Report
import layout_transactions as t
cc = columns_contain

# ReductionKey = collections.namedtuple('ReductionKey', 'year month')
# ReductionValue = collections.namedtuple('ReductionValue', 'count mean median standarddeviation')


def make_reduction_key(yyyy, mm):
    return yyyy * 100 + mm


def make_reduction_key1(yyyymm):
    return yyyymm


def make_reduction_value(prices):
    return {
        'count': len(prices),
        'mean': np.mean(prices),
        'median': np.median(prices),
        'standarddeviation': np.std(prices),
    }


def make_control(argv):
    # return a Bunch

    print argv
    parser = argparse.ArgumentParser()
    parser.add_argument('invocation')
    parser.add_argument('--data', help='reduce input and create data file in WORKING', action='store_true')
    parser.add_argument('--test', help='set internal test flag', action='store_true')
    arg = Bunch.from_namespace(parser.parse_args(argv))
    base_name = arg.invocation.split('.')[0]

    random_seed = 123
    random.seed(random_seed)

    dir_working = Path().dir_working()

    debug = False

    reduced_file_name = 'data.pickle'

    # assure output directory exists
    dir_path = dir_working + base_name + '/'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return Bunch(
        arg=arg,
        base_name=base_name,
        debug=debug,
        path_in_samples=dir_working + 'samples-train-validate.csv',
        path_out_graph=dir_path + 'median-price.pdf',
        path_out_stats_all=dir_path + 'price-stats-all.txt',
        path_out_stats_2006_2008=dir_path + 'price-stats-2006-2008.txt',
        path_reduction=dir_path + reduced_file_name,
        random_seed=random_seed,
        test=arg.test,
    )


def make_chart_graph(data, control):
    years = (2003, 2004, 2005, 2006, 2007, 2008, 2009)
    months = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    months_2009 = (1, 2, 3)
    year_month = ['%s-%02d' % (year, month)
                  for year in years
                  for month in (months_2009 if year == 2009 else months)
                  ]
    x = range(len(year_month))
    y = []
    for year in years:
        for month in (months_2009 if year == 2009 else months):
            y.append(data[make_reduction_key(year, month)]['median'])
    plt.plot(x, y)
    x_ticks = [year_month[i] if i % 12 == 0 else ' '
               for i in xrange(len(year_month))
               ]
    plt.xticks(range(len(year_month)),
               x_ticks,
               # pad=8,
               size='xx-small',
               rotation=-30,
               # rotation='vertical',
               )
    plt.yticks(size='xx-small')
    plt.xlabel('year-month')
    plt.ylabel('median price ($)')
    plt.ylim([0, 700000])
    plt.savefig(control.path_out_graph)

    plt.close()


def make_chart_stats(data, control, filter):
    'return Report with statistics for years and months that obey the filter'
    r = Report()
    format_header = '%9s %9s %9s %9s %9s %9s'
    format_detail = '%9d %9d %9.0f %9.0f %9d %9.0f'
    r.append('Prices by Month')
    r.append('')
    r.append(format_header % (' ', ' ', 'mean', 'median', 'number', 'standard'))
    r.append(format_header % ('year', 'month', 'price', 'price', 'trades', 'deviation'))
    for year in xrange(2003, 2010):
        for month in xrange(1, 13):
            if filter(year, month):
                value = data[make_reduction_key(year, month)]
                r.append(format_detail % (
                    year,
                    month,
                    value['mean'],
                    value['median'],
                    value['count'],
                    value['standarddeviation'],
                ))
    return r


def make_chart_stats_all(data, control):
    def filter(year, month):
        if year == 2009:
            return 1 <= month <= 3
        else:
            return True

    r = make_chart_stats(data, control, filter)
    r.write(control.path_out_stats_all)


def make_chart_stats_2006_2008(data, control):
    def filter(year, month):
        return year in (2006, 2007, 2008)

    r = make_chart_stats(data, control, filter)
    r.write(control.path_out_stats_2006_2008)


def make_data(control):
    transactions = pd.read_csv(control.path_in_samples,
                               nrows=1000 if control.test else None,
                               )
    # accumulate prices of transactions in each period into dict
    # key = yyyymm (the period), an
    # value = list of prices
    prices = collections.defaultdict(list)
    for index, row in transactions.iterrows():
        yyyymm = row[t.yyyymm]  # an int
        prices[yyyymm].append(row[t.price])

    # build dictionary of statistics of prices in each period
    # key = yyyymm (an int)
    # value = ReductionValue(...)
    reduction = {}
    for yyyymm, prices in prices.iteritems():
        reduction[make_reduction_key1(yyyymm)] = make_reduction_value(prices)
    return reduction


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(base_name=control.base_name)
    print control

    if control.arg.data:
        data = make_data(control)
        with open(control.path_reduction, 'wb') as f:
            pickle.dump((data, control), f)
    else:
        with open(control.path_reduction, 'rb') as f:
            data, reduction_control = pickle.load(f)
        make_chart_stats_2006_2008(data, control)
        make_chart_stats_all(data, control)
        make_chart_graph(data, control)

    print control
    if control.test:
        print 'DISCARD OUTPUT: test'
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
