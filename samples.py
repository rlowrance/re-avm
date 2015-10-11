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

from __future__ import division

import cPickle as pickle
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
    print 'usage  : python samples.py [--test]'
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

    out_file_name_base = ('-testing-' if arg.test else '') + arg.base_name

    return Bunch(
        arg=arg,
        debug=debug,
        fraction_test=0.2,
        max_sale_price=85e6,  # according to Wall Street Journal
        path_in=dir_working + 'transactions-al-g-sfr.csv',
        path_out_info_reasonable=dir_working + out_file_name_base + '-info-reasonable.pickle',
        path_out_test=dir_working + out_file_name_base + '-test.csv',
        path_out_train=dir_working + out_file_name_base + '-train.csv',
        path_out_train_test=dir_working + out_file_name_base + '-train-validate.csv',
        path_out_train_train=dir_working + out_file_name_base + '-train-train.csv',
        random_seed=random_seed,
        test=arg.test,
    )


def report_and_remove(df, keep_masks):
    'return new dataframe with just the kept rows AND info in the table that is printed'
    print 'impact of individual masks'
    format = '%40s removed %6d samples (%3d%%)'
    info = {}
    sorted_names = sorted([name for name in keep_masks.keys()])
    for name in sorted_names:
        keep_mask = keep_masks[name]
        n_removed = len(df) - sum(keep_mask)
        fraction_removed = n_removed / len(df)
        print format % (name, n_removed, 100.0 * fraction_removed)
        info[name] = fraction_removed

    mm = reduce(lambda a, b: a & b, keep_masks.values())
    total_removed = len(df) - sum(mm)
    total_fraction_removed = total_removed / len(df)
    print format % ('*** in combination', total_removed, 100.0 * total_fraction_removed)

    r = df[mm]
    return r, info


def check_never_missing(df, feature_names):
    'verify that each ege feature is always present'
    print
    print 'Each of these fields should never be missing'
    total_missing = 0
    format_string = 'field %40s is missing %7d times'
    for name in feature_names:
        count_missing = sum(pd.isnull(df[name]))
        print format_string % (name, count_missing)
        total_missing += count_missing
    print format_string % ('** total across fields **', total_missing)
    print 'total missing', total_missing
    if total_missing > 0:
        pdb.set_trace()  # should not be any missing
        pass


def check_no_zeros(df, feature_names):
    'check that only a few expected fields have zero values'
    # TODO: some of the features can have zero values!
    total_zeros = 0
    format_string = 'field %40s is zero %7d times'
    for name in feature_names:
        if ('_has_' in name) or ('is_' in name):
            continue  # these are 0/1 indicator features
        if name in (layout.building_bedrooms,
                    layout.building_basement_square_feet,
                    layout.building_fireplace_number,
                    layout.census2000_fraction_owner_occupied,
                    layout.has_pool,
                    layout.parking_spaces,):
            continue  # also, these should be often zero
        count_zero = sum(df[name] == 0)
        print format_string % (name, count_zero)
        total_zeros += count_zero
    print format_string % ('** total across fields **', total_zeros)
    if total_zeros > 0:
        print 'found some unexpected zero values'
        pdb.set_trace()  # should not be any zeros


def check_feature_values(df):
    # age features are added just before a model is fitted
    feature_names = sorted([x for x, y in Features().ege() if 'age' not in x])
    check_never_missing(df, feature_names)
    check_no_zeros(df, feature_names)


def reasonable_feature_values(df, control):
    'return new DataFrame containing sample in df that have "reasonable" values'
    def below(percentile, series):
        quantile_value = series.quantile(percentile / 100.0)
        r = series < quantile_value
        return r

    # set mask value in m to True to keep the observation
    m = {}
    m['assessment total > 0'] = df[layout.assessment_total] > 0
    m['assessment land > 0'] = df[layout.assessment_land] > 0
    m['assessment improvement > 0'] = df[layout.assessment_improvement] > 0
    m['baths > 0'] = df[layout.building_baths] > 0
    m['effective year built >= year built'] = df[layout.year_built_effective] >= df[layout.year_built]
    m['full price'] = layout.mask_full_price(df)
    m['latitude known'] = layout.mask_gps_latitude_known(df)
    m['longitude known'] = layout.mask_gps_longitude_known(df)
    m['land size < 99th percentile'] = below(99, df[layout.lot_land_square_feet])
    m['land size > 0'] = df[layout.lot_land_square_feet] > 0
    m['living size < 99th percentile'] = below(99, df[layout.building_living_square_feet])
    m['living square feet > 0'] = df[layout.building_living_square_feet] > 0
    m['median household income > 0'] = df[layout.census2000_median_household_income] > 0
    m['new or resale'] = layout.mask_new_or_resale(df)
    m['one building'] = layout.mask_is_one_building(df)
    m['one APN'] = layout.mask_is_one_parcel(df)
    # m['recording date present'] = ~df[layout.recording_date + '_deed'].isnull()  # ~ => not
    m['price > 0'] = df[layout.price] > 0
    m['price < max'] = df[layout.price] < control.max_sale_price
    m['rooms > 0'] = df[layout.building_rooms] > 0
    m['resale or new construction'] = (
        layout.mask_is_new_construction(df) |
        layout.mask_is_resale(df)
    )
    m['sale date present'] = layout.mask_sale_date_present(df)
    m['sale date valid'] = layout.mask_sale_date_valid(df)
    m['stories > 0'] = df[layout.building_stories] > 0
    m['units == 1'] = df[layout.n_units] == 1
    m['year_built > 0'] = df[layout.year_built] > 0

    print 'effects of reasonable values'
    return report_and_remove(df, m)


def add_features(df, control):
    '''mutate df

    NOTE: adding features from the 2009 taxroll leaks the future into the transaction
    set. However, the damage is perhaps small because most properties do not change
    their character: a single family house in 2009 was very likely a single family
    house in 2003
    '''
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

    # create sale-date related features
    def append_column(name, values):
        df.insert(len(df.columns),
                  name,
                  pd.Series(values, index=df.index),
                  )

    sale_date = df[layout.sale_date]
    append_column(layout.sale_date_python, sale_date.apply(python_date))
    append_column(layout.yyyymm, sale_date.apply(yyyymm))

    # create indicator features:w
    append_column(layout.is_new_construction, layout.mask_is_new_construction(df))
    append_column(layout.is_resale, layout.mask_is_resale(df))
    append_column(layout.building_has_basement, df[layout.building_basement_square_feet] > 0)
    append_column(layout.building_has_fireplace, df[layout.building_fireplace_number] > 0)
    append_column(layout.has_parking, df[layout.parking_spaces] > 0)
    append_column(layout.has_pool, df[layout.pool_flag] == 'Y')

    # create additional indicators aggregating certain PROPN codes
    def create(new_column_base, ored_column_bases):

        def create2(prefix):
            def ored_name(ored_index):
                return prefix + '_has_' + ored_column_bases[ored_index]

            mask = df[ored_name(0)]
            for index in range(1, len(ored_column_bases)):
                mask2 = df[ored_name(index)]
                mask = mask | mask2
            append_column(prefix + '_has_' + new_column_base, mask)

        for prefix in ('census_tract', 'zip5'):
            create2(prefix)

    create('any_commercial', ('commercial', 'commercial_condominium',))
    create('any_industrial', ('industrial', 'industrial_light', 'industrial_heavy',))
    create('any_non_residential', ('amusement', 'any_commercial', 'financial_institution', 'hotel',
                                   'any_industrial', 'medical', 'office_building', 'parking',
                                   'retail', 'service', 'transport', 'utilities', 'warehouse',))


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(base_name=control.arg.base_name)
    print control

    transactions = pd.read_csv(control.path_in,
                               nrows=100000 if control.arg.test else None,
                               )
    print 'transactions shape', transactions.shape

    after_2000_census_known = transactions[layout.mask_sold_after_2002(transactions)]
    print 'after 2000 census known shape', after_2000_census_known.shape

    subset, info_reasonable = reasonable_feature_values(after_2000_census_known, control)
    print 'subset shape', subset.shape

    add_features(subset, control)
    check_feature_values(subset)

    # split into test and train
    # stratify by yyyymm (month of sale)

    def count_yyyymm(df, yyyymm):
        return sum(df[layout.yyyymm] == yyyymm)

    def split(df, fraction_test):
        sss = cross_validation.StratifiedShuffleSplit(
            y=df.yyyymm,
            n_iter=1,
            test_size=fraction_test,
            train_size=None,
            random_state=control.random_seed,
        )
        assert len(sss) == 1
        for train_index, test_index in sss:
            train = df.iloc[train_index]
            test = df.iloc[test_index]
        return test, train

    # split samples into test and train, stratified on month of sale
    # then split training data on test (validate) and train
    test, train = split(subset, control.fraction_test)
    fraction_test = control.fraction_test / (1 - control.fraction_test)
    train_test, train_train = split(train, fraction_test)

    # write the csv files
    test.to_csv(control.path_out_test)
    train.to_csv(control.path_out_train)
    train_test.to_csv(control.path_out_train_test)
    train_train.to_csv(control.path_out_train_train)

    # count samples in each strata (= month)
    yyyymms = sorted(set(subset[layout.yyyymm]))
    format_string = '%6d # total %6d # test %6d # train %6d # train_test %6d # train_train %6d'
    for yyyymm in yyyymms:
        c1 = count_yyyymm(subset, yyyymm)
        c2 = count_yyyymm(test, yyyymm)
        c3 = count_yyyymm(train, yyyymm)
        c4 = count_yyyymm(train_test, yyyymm)
        c5 = count_yyyymm(train_train, yyyymm)
        print format_string % (yyyymm, c1, c2, c3, c4, c5)
        if c1 != c2 + c3:
            print 'c1 not exactly split'
            pdb.set_trace()
        if c3 != c4 + c5:
            print 'c3 not exactly split'
            pdb.set_trace()
    print 'totals'
    print format_string % (0, len(subset), len(test), len(train), len(train_train), len(train_test))

    f = open(control.path_out_info_reasonable, 'wb')
    pickle.dump((info_reasonable, control), f)
    f.close()

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
