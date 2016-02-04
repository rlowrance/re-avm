'''create charts showing median and mean prices each month

INVOCATION
  python chart-01.py [--data] [--test]

INPUT FILES
 INPUT/samples-train-validate.csv

OUTPUT FILES
 WORKING/chart-01/data.pickle
 WORKING/chart-01/txt
 WORKING/chart-01/pdf
'''

from __future__ import division

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
from ParseCommandLine import ParseCommandLine
from Path import Path
from Report import Report
import layout_transactions as t
cc = columns_contain


def usage(msg=None):
    print __doc__
    if msg is not None:
        print msg
    sys.exit(1)


def make_control(argv):
    # return a Bunch

    print argv
    if len(argv) not in (1, 2, 3):
        usage('invalid number of arguments')

    pcl = ParseCommandLine(argv)
    arg = Bunch(
        base_name='chart-01',
        data=pcl.has_arg('--data'),
        test=pcl.has_arg('--test'),
    )

    random_seed = 123
    random.seed(random_seed)

    dir_working = Path().dir_working()

    debug = False

    reduced_file_name = ('test-' if arg.test else '') + 'data.pickle'

    # assure output directory exists
    dir_path = dir_working + arg.base_name + '/'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return Bunch(
        arg=arg,
        debug=debug,
        path_in_samples=dir_working + 'samples-train-validate.csv',
        path_out_pdf=dir_path + 'median-price.pdf',
        path_out_txt=dir_path + 'median-price.txt',
        path_out_txt_2006_2007=dir_path + 'median-price_2006_2007.txt',
        path_reduction=dir_path + reduced_file_name,
        random_seed=random_seed,
        test=arg.test,
    )


def key(year, month):
    return year * 100 + month


def make_chart_pdf(data, control):
    counts, means, medians, prices = data
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
            y.append(medians[key(year, month)])
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
    plt.savefig(control.path_out_pdf)

    plt.close()


def make_chart_txt(data, control):
    counts, means, medians, prices = data
    r = Report()
    format_header = '%7s %7s %7s %7s %7s'
    format_detail = '%7d %7d %7.0f %7.0f %7d'
    r.append('Prices by Month')
    r.append('')
    r.append(format_header % (' ', ' ', 'mean', 'median', 'number'))
    r.append(format_header % ('year', 'month', 'price', 'price', 'trades'))
    for year in xrange(2003, 2010):
        for month in (xrange(1, 4) if year == 2009 else xrange(1, 13)):
            k = key(year, month)
            r.append(format_detail % (
                year,
                month,
                means[k],
                medians[k],
                counts[k],
            ))
    r.write(control.path_out_txt)


def make_chart_txt_2006_2007(data, control):
    counts, means, medians, prices = data
    r = Report()
    format_header = '%7s %7s %7s %7s'
    format_detail = '%7d %7d %7.0f %7d'
    r.append('Median Prices and Transaction Counts by Month')
    r.append('')
    r.append(format_header % (' ', ' ', 'median', 'number'))
    r.append(format_header % ('year', 'month', 'price', 'trades'))
    for year in (2006, 2007):
        for month in xrange(1, 13):
            k = key(year, month)
            r.append(format_detail % (
                year,
                month,
                medians[k],
                counts[k],
            ))
    r.write(control.path_out_txt_2006_2007)


def make_data(control):
    transactions = pd.read_csv(control.path_in_samples,
                               nrows=1000 if control.test else None,
                               )
    prices = collections.defaultdict(list)
    for index, row in transactions.iterrows():
        yyyymm = row[t.yyyymm]
        prices[yyyymm].append(row[t.price])
    counts = {}
    means = {}
    medians = {}
    for k, v in prices.iteritems():
        a = np.array(v)
        counts[k] = len(a)
        means[k] = np.mean(a)
        medians[k] = np.median(a)
    return counts, means, medians, prices


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(base_name=control.arg.base_name)
    print control

    if control.arg.data:
        data = make_data(control)
        with open(control.path_reduction, 'wb') as f:
            pickle.dump((data, control), f)
    else:
        with open(control.path_reduction, 'rb') as f:
            data, reduction_control = pickle.load(f)
        make_chart_txt_2006_2007(data, control)
        make_chart_txt(data, control)
        make_chart_pdf(data, control)

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
