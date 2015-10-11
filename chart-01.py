'''create charts showing median and mean prices each month

INPUT FILES
 INPUT/samples-train-validate.csv

OUTPUT FILES
 WORKING/chart-01.data.pickle
 WORKING/chart-01.txt
'''

from __future__ import division

import collections
import cPickle as pickle
import numpy as np
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
    if msg is not None:
        print msg
    print 'usage  : python chart-01.py [--data] [--test]'
    print ' --data: produce reduction of the input file, not the actual charts'
    print ' --test: run in test mode'
    sys.exit(1)


def make_control(argv):
    # return a Bunch

    print argv
    if len(argv) not in (1, 2, 3):
        usage('invalid number of arguments')

    pcl = ParseCommandLine(argv)
    arg = Bunch(
        base_name=argv[0].split('.')[0],
        data=pcl.has_arg('--data'),
        test=pcl.has_arg('--test'),
    )

    random_seed = 123
    random.seed(random_seed)

    dir_working = Path().dir_working()

    debug = False

    out_file_name_base = ('testing-' if arg.test else '') + arg.base_name

    return Bunch(
        arg=arg,
        debug=debug,
        fraction_test=0.2,
        max_sale_price=85e6,  # according to Wall Street Journal
        path_in_samples=dir_working + 'samples-train-validate.csv',
        path_out_txt=dir_working + out_file_name_base + '.txt',
        path_data=dir_working + out_file_name_base + '.data.pickle',
        random_seed=random_seed,
        test=arg.test,
    )


DataKey = collections.namedtuple('DataKey', 'year month')


def make_chart_txt(data, control):
    counts, means, medians, prices = data
    r = Report()
    format_header = '%7s %7s %7s %7s %7s'
    format_detail = '%7d %7d %7.0f %7.0f %7d'
    r.append('Chart 01: Prices by Month')
    r.append('')
    r.append(format_header % (' ', ' ', 'mean', 'median', 'number'))
    r.append(format_header % ('year', 'month', 'price', 'price', 'trades'))
    for year in xrange(2003, 2010):
        for month in (xrange(1, 4) if year == 2009 else xrange(1, 13)):
            key = DataKey(year=year, month=month)
            r.append(format_detail % (
                year,
                month,
                means[key],
                medians[key],
                counts[key],
            ))
    return r


def make_data(control):
    transactions = pd.read_csv(control.path_in_samples,
                               nrows=1000 if control.test else None,
                               )
    prices = collections.defaultdict(list)
    for index, row in transactions.iterrows():
        yyyymm = row[t.yyyymm]
        yyyy = int(yyyymm / 100.0)
        mm = int(yyyymm % 100.0)
        prices[DataKey(year=yyyy, month=mm)].append(row[t.price])
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
        with open(control.path_data, 'wb') as f:
            pickle.dump((data, control), f)
    else:
        with open(control.path_data, 'rb') as f:
            data, control = pickle.load(f)

    r = make_chart_txt(data, control)
    r.write(control.path_out_txt)

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
