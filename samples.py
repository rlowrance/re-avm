'''create training and test samples from the transactions
* add certain fields
* select transactions that contain "reasonable" values

INPUT FILES
 INPUT/transactions-al-g-sfr.csv

OUTPUT FILES
 WORKING/samples-test.csv
 WORKING/samples-train.csv
 WORKING/samples-train-validate.csv
 WORKING/samples-validate.csv
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
from columns_contain import columns_contain
from Features import Features
from Logger import Logger
from ParseCommandLine import ParseCommandLine
from Path import Path
import layout_transactions as layout
cc = columns_contain


def usage(msg=None):
    if msg is not None:
        print msg
    print 'usage  : python transactions_subset.py [--test]'
    print ' --test: run in test mode'
    sys.exit(1)


def make_control(argv):
    # return a Bunch

    print argv
    if len(argv) not in (1, 2):
        usage('invalid number of arguments')

    pcl = ParseCommandLine(argv)
    arg = Bunch(
        base_name=argv[0].split('.')[0],
        test=pcl.has_arg('--test'),
    )

    random_seed = 123
    random.seed(random_seed)

    dir_working = Path().dir_working()

    debug = False

    return Bunch(
        arg=arg,
        debug=debug,
        fraction_test=0.1,
        max_sale_price=85e6,  # according to Wall Street Journal
        path_in=dir_working + 'transactions-al-g-sfr.csv',
        path_out_test=dir_working + arg.base_name + '-test.csv',
        path_out_train=dir_working + arg.base_name + '-train.csv',
        path_out_train_validate=dir_working + arg.base_name + '-train-validate.csv',
        path_out_validate=dir_working + arg.base_name + '-validate.csv',
        random_seed=random_seed,
        test=arg.test,
    )


def report_and_remove(df, keep_masks):
    pdb.set_trace()
    print 'impact of individual masks'
    format = '%30s removed %6d samples (%3d%%)'
    for name, keep_mask in keep_masks.iteritems():
        n_removed = len(df) - sum(keep_mask)
        fraction_removed = n_removed / len(df)
        print format % (name, n_removed, 100.0 * fraction_removed)

    pdb.set_trace()
    mm = reduce(lambda a, b: a & b, keep_masks.values())
    total_removed = len(df) - sum(mm)
    total_fraction_removed = total_removed / len(df)
    print format % ('*** in combination', total_removed, 100.0 * total_fraction_removed)

    r = df[mm]
    return r


def always_present_ege_features(df, control):
    '''return those rows in the df that have no missing values needed for the ege analysis

    ege = estimated generalization error; see program ege_week.py for an example
    '''
    pdb.set_trace()
    m = {}
    for name, _ in Features().ege():
        print name
        if name not in df.columns:
            # expect age-related feature to be missing
            # these features are computed when the model is fit
            # expect has_pool, etc. to be missing
            # these features are computed in the add_fields function in this module
            print name, 'not in df'
            if name in (layout.age, layout.age2, layout.age_effective, layout.age_effective2):
                # these fields are determined when the model is run
                # they depend in the sale date being studied
                pass
            elif name in (layout.building_has_pool, layout.building_is_new_construction):
                # these fields are computed in this module in the add_fields function
                pass
            elif name not in df.columns:
                print 'df columns', df.columns
                print name, 'not in df'
                pdb.set_trace()
            else:
                m[name] = pd.notnull(df[name])
    pdb.set_trace()
    print 'effects of always present in ege individually'
    return report_and_remove(df, m)


def reasonable_feature_values(df, control):
    def below(percentile, series):
        quantile_value = series.quantile(percentile / 100.0)
        r = series < quantile_value
        return r

    # set mask value in m to True to keep the observation
    m = {}
    m['one building'] = layout.mask_is_one_building(df)
    m['one APN'] = layout.mask_is_one_parcel(df)
    m['assessment total > 0'] = df[layout.assessment_total] > 0
    m['assessment land > 0'] = df[layout.assessment_land] > 0
    m['assessment improvement > 0'] = df[layout.assessment_improvement] > 0
    m['effective_year_built > 0'] = df[layout.year_built_effective] > 0
    m['year_built > 0'] = df[layout.year_built] > 0
    m['effective year >= year built'] = df[layout.year_built_effective] >= df[layout.year_built]
    m['latitude known'] = layout.mask_gps_latitude_known(df)
    m['longitude known'] = layout.mask_gps_longitude_known(df)
    m['land size < 99th percentile'] = below(99, df[layout.lot_land_square_feet])
    m['living size < 99th percentile'] = below(99, df[layout.building_living_square_feet])
    # m['recording date present'] = ~df[layout.recording_date + '_deed'].isnull()  # ~ => not
    m['price > 0'] = df[layout.price] > 0
    m['price < max'] = df[layout.price] < control.max_sale_price
    m['full price'] = layout.mask_full_price(df)
    m['rooms > 0'] = df[layout.building_rooms] > 0
    m['new or resale'] = layout.mask_new_or_resale(df)
    m['units == 1'] = df[layout.n_units] == 1
    m['sale date present'] = layout.mask_sale_date_present(df)
    m['sale date valid'] = layout.mask_sale_date_valid(df)

    print 'effects of reasonable values'
    return report_and_remove(df, m)


def add_features(df, control):
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

    # TODO: add has_pool


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(base_name=control.arg.base_name)
    print control

    transactions = pd.read_csv(control.path_in,
                               nrows=100000 if control.arg.test else None,
                               )
    print 'transactions shape', transactions.shape

    pdb.set_trace()
    after_2000_census_known = transactions[layout.mask_sold_after_2002(transactions)]
    print 'after 2000 census known shape', after_2000_census_known.shape
    reasonable = reasonable_feature_values(after_2000_census_known, control)
    print 'reasonable shape', reasonable.shape
    subset = always_present_ege_features(reasonable, control)
    print 'subset shape', subset.shape

    add_features(subset, control)

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
        pd.DataFrame()
        np.array()

    main(sys.argv)
