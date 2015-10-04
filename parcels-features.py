'''create census tract and zip5 features of each parcel

INPUT FILES
 INPUT/corelogic-deeds-*/CAC*.txt

OUTPUT FILES
 WORKING/parcels-features-census-tract-sfr.csv
 WORKING/parcels-features-zip5-sfr.csv

Each parcels was classified as single family retail.

The fields in the output csv files are
 census_tract|zip5  primary key
 has_commercial
 has_industry
 has_park
 has_retail
 has_school

The output files are joined to the subset.

Joining them to transactions.csv ran out of memory with 64 GB
'''

import numpy as np
import pandas as pd
import pdb
from pprint import pprint
import random
import sys

from Bunch import Bunch
import layout_parcels as parcels
from Logger import Logger
from Path import Path
from ParseCommandLine import ParseCommandLine


def usage(msg=None):
    if msg is not None:
        print msg
    print 'usage  : python parcels-features.py --geo GEO [--test]'
    print ' GEO   : either census_tract or zip5'
    print ' --test: run in test mode'
    sys.exit(1)


def make_control(argv):
    # return a Bunch

    print argv
    if len(argv) not in (3, 4):
        usage('invalid number of arguments')

    pcl = ParseCommandLine(argv)
    arg = Bunch(
        base_name=argv[0].split('.')[0],
        geo=pcl.get_arg('--geo'),
        test=pcl.has_arg('--test'),
    )
    if arg.geo is None:
        usage('missing --arg')
    if arg.geo not in ('census_tract', 'zip5'):
        usage('invalid GEO value: ', + arg.geo)

    random_seed = 123456
    random.seed(random_seed)

    path = Path()  # use the default dir_input

    debug = False

    return Bunch(
        arg=arg,
        debug=debug,
        max_sale_price=85e6,  # according to Wall Street Journal
        path=path,
        path_out=path.dir_working() + arg.base_name + '-' + arg.geo + '.csv',
        random_seed=random_seed,
        test=arg.test,
    )


def derive(parcels_df, parcels_geo_column_name, parcels_mask_function,
           transactions_df, transactions_geo_column_name, new_feature_name):
    'add new_feature_name to transactions_df'

    # get the unique geo ids
    mask = parcels_mask_function(parcels_df)
    parcels_df_subset = parcels_df[mask]
    geo_ids_all = parcels_df_subset[parcels_geo_column_name]
    geo_ids = set(int(geo_id)
                  for geo_id in geo_ids_all
                  if not np.isnan(geo_id)
                  )
    print parcels_geo_column_name, len(geo_ids), parcels_mask_function

    if len(geo_ids) == 0:
        # this happens turning testing and maybe during production
        transactions_df[new_feature_name] = pd.Series(data=False, index=transactions_df.index)
    else:
        all_indicators = []

        print 'number of indicators for', parcels_geo_column_name, parcels_mask_function
        for geo_id in geo_ids:
            indicators = transactions_df[transactions_geo_column_name] == geo_id
            print transactions_geo_column_name, geo_id, sum(indicators)
            all_indicators.append(indicators)
        has_feature = reduce(lambda a, b: a | b, all_indicators)
        print transactions_geo_column_name, 'all', sum(has_feature)
        transactions_df[new_feature_name] = pd.Series(has_feature, index=transactions_df.index)
        print 'new feature %25s is True %6d times' % (
            new_feature_name, sum(transactions_df[new_feature_name]))


def test_derived():
    'unit test'
    verbose = False

    def vp(x):
        if verbose:
            print x

    parcels_df = pd.DataFrame({'geo': [1, 2, 3]})

    def parcels_mask_function(parcels_df):
        return parcels_df['geo'] >= 2

    transactions_df = pd.DataFrame({'geo_t': [1, 2, 3, 2, 1]})
    vp(parcels_df)
    vp(transactions_df)

    derive(parcels_df, 'geo', parcels_mask_function, transactions_df, 'geo_t', 'new feature')
    new_column = transactions_df['new feature']
    vp(new_column)
    assert new_column.equals(pd.Series([False, True, True, True, False])), new_column


def parcels_derived_features(parcels_df, target_df):
    'return census and zip5 dataframes, identifying special geo areas in target_df'
    def truncate_zipcode(zip):
        'convert possible zip9 to zip5'
        x = zip / 10000.0 if zip > 99999 else zip
        return int(x if not np.isnan(x) else 0)

    # some zipcodes are 5 digits, others are 9 digits
    # create new feature that has the first 5 digits of the zip code
    def add_zip5(df, zip9_column_name, zip5_column_name):
        df[parcels.zip5] = df[parcels.zip9].apply(truncate_zipcode)
    add_zip5(parcels_df)
    add_zip5(target_df)

    def make_geo_ids(geo, mask_function):
        def make(column_name):
            print geo, mask_function, column_name
            parcels_df_subset = parcels_df[mask_function(parcels_df)]
            items = parcels_df_subset[column_name]
            r = set(int(item)
                    for item in items
                    if not np.isnan(item)
                    )
            return r
        if geo == 'census_tract':
            return make(parcels.census_tract)
        else:
            return make(parcels.zip5)

    name_masks = (
        ('has_commercial', parcels.mask_commercial),
        ('has_industry', parcels.mask_industry),
        ('has_park', parcels.mask_park),
        ('has_retail', parcels.mask_retail),
        ('has_school', parcels.mask_school),
    )
    geo_names = (parcels.census_tract, parcels.zip5)

    for geo_name in geo_names:
        for mask_name, mask_function in name_masks:
            pass
#           new_feature_name = geo_name + ' ' + mask_name
#            derive(parcels_df, parcels_column_name, mask_function,
#                   transactions_df, transactions_column_name, new_feature_name)


def just_used(geo, df):
    'return new DataFrame containing just columns we need for further processing'
    r = pd.DataFrame({
        'geo': df[parcels.census_tract] if geo == 'census_tract' else df[parcels.zip5],
        parcels.land_use: df[parcels.land_use],
        parcels.property_indicator: df[parcels.property_indicator],
    })
    return r


def make_indicator(df, indicator_mask_function):
    'return a Series'
    pdb.set_trace()
    d = df[indicator_mask_function(df)].values
    print d
    pass


def make_zip5(x):
    pdb.set_trace()
    xx = np.array(x / 10000.0, dtype='int32')
    xx[xx < 0] = 0
    return xx


def make_has_indicatorsOLD(df, name_masks):
    'return new DataFrame'
    result = pd.DataFrame()
    pdb.set_trace()
    for indicator_name, indicator_mask_function in name_masks:
        mask = indicator_mask_function(df)
        indicated_geos = df[mask]['geo']
        geo_ids = set(indicated_geos)
        all_indicators = []
        geo_column = df['geo']
        for n, geo_id in enumerate(geo_ids, start=1):
            print 'geo_id %d of %d: %d' % (n, len(geo_ids),  geo_id)
            geo_id_indicators = geo_column == geo_id
            all_indicators.append(geo_id_indicators)
        new_column = reduce(lambda a, b: a | b, all_indicators)
        result[indicator_name] = new_column
    pdb.set_trace()
    return result


def make_has_indicators(df, name_masks):
    'return new df with on index for each geo value'
    result_index = set(df.index)
    result = {}
    for name, mask in name_masks:
        is_feature = df[mask(df)]
        result[name] = pd.Series(data=[False] * len(result_index),
                                 index=result_index)
        for is_true in set(is_feature.index):
            print name, is_true
            result[name][is_true] = True
    r = pd.DataFrame(data=result)
    return r


def test_make_has_indicators():
    df = pd.DataFrame({'geo': (1, 2, 3, 2, 1),
                       'x': (11, 12, 13, 13, 15),  # yes, 2 x 13
                       })
    df = pd.DataFrame(data={'x': (11, 12, 13, 13, 15),  # yes, 2 x 13
                            },
                      index=(1, 2, 3, 2, 1),
                      )

    def mask_is_even(df):
        r = df['x'] % 2 == 0
        return r

    def mask_is_odd(df):
        return ~mask_is_even(df)

    name_masks = (('has_even', mask_is_even),
                  ('has_odd', mask_is_odd),
                  )
    r = make_has_indicators(df, name_masks)
    print r
    assert len(r) == 3
    assert (r.has_even == [False, True, False]).all()
    assert (r.has_odd == [True, True, True]).all()


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(base_name=control.arg.base_name)
    print control

    # create dataframes
    parcels_df = parcels.read(control.path,
                              10000 if control.test else None)
    print 'parcels df shape', parcels_df.shape

    # drop samples without the geographic indicator we will use
    # add zip5 field
    if control.arg.geo == 'zip5':
        parcels_df = parcels_df[parcels.mask_parcel_has_zipcode(parcels_df)]
        parcels_df[parcels.zip5] = pd.Series(data=parcels_df[parcels.zipcode] / 10000.0,
                                             dtype=np.int32,
                                             index=parcels_df.index)
    elif control.arg.geo == 'census_tract':
        # drop if no census tract
        parcels_df = parcels_df[parcels.mask_parcel_has_census_tract(parcels_df)]
    else:
        print 'bad control.arg.geo', control.arg.geo
        pdb.set_trace()

    # the computation runs out of memory on 64GB if all columns are retained
    # so drop all but the columns needed
    parcels_df = just_used(control.arg.geo, parcels_df)
    parcels_sfr_df = parcels_df[parcels.mask_is_sfr(parcels_df)]

    print 'parcels sfr df shape', parcels_sfr_df.shape

    # indicator_df = parcels_derived_features(geo_column_name, parcels_selected_df)
    name_masks = (
        ('has_commercial', parcels.mask_is_commercial),
        ('has_industry', parcels.mask_is_industry),
        ('has_park', parcels.mask_is_park),
        ('has_retail', parcels.mask_is_retail),
        ('has_school', parcels.mask_is_school),
    )
    parcels_df.index = parcels_df.geo  # the index must be the geo field
    has_indicators = make_has_indicators(parcels_df, name_masks)
    print 'has_indicators shape', has_indicators.shape
    if control.test:
        print has_indicators
    has_indicators.to_csv(control.path_out)

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

    test_make_has_indicators()
    main(sys.argv)
