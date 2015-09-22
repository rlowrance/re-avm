'''create subset of the transactions file version 3; add fields

INPUT FILES
 INPUT/transactions3-al-g-sfr.csv

OUTPUT FILES
 WORKING/transactions-subset3-test.csv
 WORKING/transactions-subset3-train.csv
'''

import datetime
import numpy as np
import pandas as pd
import pdb
from pprint import pprint
import random
from sklearn import cross_validation
import sys

from Bunch import Bunch
from directory import directory
from Logger import Logger
import parse_command_line
import transactions3_subset_layout as layout


def usage(msg=None):
    if msg is not None:
        print msg
    print 'usage  : python transactions_subset3.py [--test]'
    print ' --test: run in test mode'
    sys.exit(1)


def make_control(argv):
    # return a Bunch

    print argv
    if len(argv) not in (1, 2):
        usage('invalid number of arguments')

    pcl = parse_command_line.ParseCommandLine(argv)
    arg = Bunch(
        base_name=argv[0].split('.')[0],
        test=pcl.has_arg('--test'),
    )

    random_seed = 123
    random.seed(random_seed)

    debug = False

    return Bunch(
        arg=arg,
        debug=debug,
        fraction_test=0.1,
        max_sale_price=85e6,  # according to Wall Street Journal
        path_in=directory('working') + 'transactions3-al-g-sfr.csv',
        path_out_test=directory('working') + arg.base_name + '-test.csv',
        path_out_train=directory('working') + arg.base_name + '-train.csv',
        random_seed=random_seed,
        test=arg.test,
    )


def reasonable_feature_values(all_samples, control):
    def below(percentile, series):
        quantile_value = series.quantile(percentile / 100.0)
        r = series < quantile_value
        return r

    debug = True
    a = all_samples
    if False and debug:
        for column_name in a.columns:
            print column_name

    # set mask value in m to True to keep the observation
    m = {}
    m['one building'] = layout.mask_is_one_building(a)
    m['one APN'] = layout.mask_is_one_parcel(a)
    m['assessment total > 0'] = a[layout.assessment_total] > 0
    m['assessment land > 0'] = a[layout.assessment_land] > 0
    m['assessment improvement > 0'] = a[layout.assessment_improvement] > 0
    m['effective_year_built > 0'] = a[layout.year_built_effective] > 0
    m['year_built > 0'] = a[layout.year_built] > 0
    m['effective year >= year built'] = a[layout.year_built_effective] >= a[layout.year_built]
    m['latitude known'] = layout.mask_gps_latitude_known(a)
    m['longitude known'] = layout.mask_gps_longitude_known(a)
    m['land size'] = below(99, a[layout.land_size])
    m['living size'] = below(99, a[layout.living_size])
    # m['recording date present'] = ~a[layout.recording_date + '_deed'].isnull()  # ~ => not
    m['price > 0'] = a[layout.price] > 0
    m['price < max'] = a[layout.price] < control.max_sale_price
    m['full price'] = layout.mask_full_price(a)
    m['rooms > 0'] = a[layout.n_rooms] > 0
    m['new or resale'] = layout.mask_new_or_resale(a)
    m['units == 1'] = a[layout.n_units] == 1
    m['sale date present'] = layout.mask_sale_date_present(a)
    m['sale date valid'] = layout.mask_sale_date_valid(a)

    print 'effect of conditions individually'
    for k, v in m.iteritems():
        removed = len(a) - sum(v)
        print '%30s removed %6d samples (%3d%%)' % (k, removed, 100.0 * removed / len(a))

    mm = reduce(lambda a, b: a & b, m.values())
    total_removed = len(a) - sum(mm)
    print 'in combination, removed %6d samples (%3d%%)' % (total_removed, 100.0 * total_removed / len(a))

    r = a[mm]
    return r


def add_fields(df, control):
    'mutate df'
    def split(date):
        year = int(date / 10000)
        md = date - year * 10000
        month = int(md / 100)
        day = md - month * 100
        return year, month, day

    def python_date(date):
        'yyyymmdd --> datetime.date(year, month, day)'
        year, month, day = split(date)
        return datetime.date(int(year), int(month), int(day))

    def yyyymm(date):
        year, month, day = split(date)
        return year * 100 + month

    value = df[layout.sale_date]

    sale_date_python = value.apply(python_date)
    df[layout.sale_date_python] = pd.Series(sale_date_python)

    yyyymm = value.apply(yyyymm)
    df[layout.yyyymm] = pd.Series(yyyymm)


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(base_name=control.arg.base_name)
    print control

    transactions = pd.read_csv(control.path_in,
                               nrows=100000 if control.arg.test else None,
                               )
    print 'transactions column names'
    for c in transactions.columns:
        print c
    print 'transactions.shape', transactions.shape

    after_2000_census_known = transactions[layout.mask_sold_after_2002(transactions)]
    print 'after 2000 census known shape', after_2000_census_known.shape
    subset = reasonable_feature_values(after_2000_census_known, control)
    print 'subset shape', subset.shape

    # add fields
    add_fields(subset, control)

    # split into test and train
    # stratify by yyyymm (month of sale)

    def count_yyyymm(df, yyyymm):
        return sum(df[layout.yyyymm] == yyyymm)

    rs = cross_validation.StratifiedShuffleSplit(y=subset[layout.yyyymm],
                                                 n_iter=1,
                                                 test_size=control.fraction_test,
                                                 train_size=None,
                                                 random_state=control.random_seed)
    assert len(rs) == 1
    for train_index, test_index in rs:
        print 'len train', len(train_index), 'len test', len(test_index)
        assert len(train_index) > len(test_index)
        train = subset.iloc[train_index]
        test = subset.iloc[test_index]

    # count samples in each strata
    yyyymms = sorted(set(subset[layout.yyyymm]))
    format = '%6d # total %6d # test %6d # train %6d'
    for yyyymm in yyyymms:
        c1 = count_yyyymm(subset, yyyymm)
        c2 = count_yyyymm(test, yyyymm)
        c3 = count_yyyymm(train, yyyymm)
        print format % (yyyymm, c1, c2, c2)
        if c1 != c2 + c3:
            print 'not exactly split'
            pdb.set_trace()
    print 'totals'
    print format % (0, len(subset), len(test), len(train))

    train.to_csv(control.path_out_train)
    test.to_csv(control.path_out_test)

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
        parse_command_line()
        pd.DataFrame()
        np.array()

    main(sys.argv)
