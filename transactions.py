'''join deeds and tax roll files to create arms-length, grant, and sfr transactions

INPUT FILES
 INPUT/corelogic-deeds-090402_07/CAC06037F1.zip ...
 INPUT/corelogic-deeds-090402_09/CAC06037F1.zip ...
 INPUT/corelogic-taxrolls-090402_05/CAC06037F1.zip ...

OUTPUT FILES
 WORKING/transactions-al-g-sfr.csv

NOTES:
    1. The deeds file has the prior sale info, which we could use
       to create more transactions. We didn't, because we only have
       census data from the year 2000, and we use census tract
       features, so its not effective to go back before sometime
       in 2002, when the 2000 census data became public.

    2. Fields created in this program
       avg_commute
       fraction_owner_occupied
       median_household_income
       X_has_commercial
       X_has_industry
       X_has_park
       X_has_retail
       X_has_school, for X in {census_tract, zip5}
       best_apn
'''

import collections
import numpy as np
import pandas as pd
import pdb
from pprint import pprint
import random
import sys


from Bunch import Bunch
import layout_census as census
import layout_deeds as deeds
import layout_parcels as parcels
import layout_transactions as transactions
from Logger import Logger
from ParseCommandLine import ParseCommandLine
from Path import Path


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

    pcl = ParseCommandLine(argv)
    arg = Bunch(
        base_name=argv[0].split('.')[0],
        test=pcl.has_arg('--test'),
    )

    random_seed = 123456
    random.seed(random_seed)

    path = Path()  # use the default dir_input

    debug = False

    return Bunch(
        arg=arg,
        debug=debug,
        max_sale_price=85e6,  # according to Wall Street Journal
        path=path,
        path_out_transactions=path.dir_working() + arg.base_name + '-al-g-sfr.csv',
        random_seed=random_seed,
        test=arg.test,
    )


def best_apn(df, feature_formatted, feature_unformatted):
    '''return series with best apn

    Algo for the R version
     use unformatted, if its all digits
     otherwise, use formatted, if removing hyphens makes a number
     otherwise, use NaN
    '''
    formatted = df[feature_formatted]
    unformatted = df[feature_unformatted]
    if False:
        print unformatted.head()
        print formatted.head()
    if np.dtype(unformatted) == np.int64:
        # every entry is np.int64, because pd.read_csv made it so
        return unformatted
    if np.dtype(unformatted) == np.object:
        return np.int64(unformatted)
    print 'not expected'
    pdb.set_trace()


def read_census(control):
    'return dataframe'
    print 'reading census'
    df = pd.read_csv(control.path.dir_input('census'), sep='\t')
    return df


def read_geocoding(control):
    'return dataframe'
    print 'reading geocoding'
    df = pd.read_csv(control.path.dir_input('geocoding'), sep='\t')
    return df


def columns_contain(x, df):
    'for interactive debugging'
    print [column_name for column_name in df.columns if x.lower() in column_name.lower()]


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


def parcels_derived_features(parcels_df, transactions_df):
    'mutate transactions_df by addinging indicator columns for tract and zip5 code features'
    def truncate_zipcode(zip):
        'convert possible zip9 to zip5'
        x = zip / 10000.0 if zip > 99999 else zip
        return int(x if not np.isnan(x) else 0)

    # some zipcodes are 5 digits, others are 9 digits
    # create new feature that has the first 5 digits of the zip code
    transactions_df[transactions.zip5] = transactions_df[transactions.zip9].apply(truncate_zipcode)
    parcels_df[parcels.zip5] = parcels_df[parcels.zip9].apply(truncate_zipcode)

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
    column_names = (
        (parcels.census_tract, transactions.census_tract),
        (parcels.zip5, transactions.zip5),
    )

    for parcels_column_name, transactions_column_name in column_names:
        for mask_name, mask_function in name_masks:
            new_feature_name = parcels_column_name + ' ' + mask_name
            derive(parcels_df, parcels_column_name, mask_function,
                   transactions_df, transactions_column_name, new_feature_name)
            continue
#            new_feature_name = geo + '_' + transactions_column_name_suffix
#            geo_ids = make_geo_ids(geo_name, mask_function)   # census tract nums or zip5 nums
#            print geo_name, transactions_column_name_suffix, len(geo_ids)
#            if len(geo_ids) == 0:
#                # this happens turning testing and maybe during production
#                transactions_df[new_feature_name] = pd.Series(data=False, index=transactions_df.index)
#            else:
#                all_indicators = []
#                for geo_id in geo_ids:
#                    indicators = transactions_df[parcels_column_name_geo] == geo_id
#                    print geo_name, parcels_column_name_geo, geo_id, sum(indicators)
#                    all_indicators.append(indicators)
#                has_feature = reduce(lambda a, b: a | b, all_indicators)
#                print geo, name, 'all', sum(has_feature)
#                transactions_df[new_feature_name] = pd.Series(has_feature, index=transactions_df.index)
#                print 'new feature %25s is True %6d times' % (
#                    new_feature_name, sum(transactions_df[new_feature_name]))
#

CensusFeatures = collections.namedtuple('CensusFeatures', (
    'avg_commute', 'median_hh_income', 'fraction_owner_occupied',
)
)


def reduce_census(census_df):
    'return dictionary[census_tract] --> CensusFeatures'

    def get_census_tract(row):
        fips_census_tract = float(row[census.fips_census_tract])
        census_tract = int(fips_census_tract % 1000000)
        return census_tract

    def get_avg_commute(row):
        'return weighted average commute time'
        def mul(factor):
            return (factor[0] * float(row[census.commute_less_5]) +
                    factor[1] * float(row[census.commute_5_to_9]) +
                    factor[2] * float(row[census.commute_10_to_14]) +
                    factor[3] * float(row[census.commute_15_to_19]) +
                    factor[4] * float(row[census.commute_20_to_24]) +
                    factor[5] * float(row[census.commute_25_to_29]) +
                    factor[6] * float(row[census.commute_30_to_34]) +
                    factor[7] * float(row[census.commute_35_to_39]) +
                    factor[8] * float(row[census.commute_40_to_44]) +
                    factor[9] * float(row[census.commute_45_to_59]) +
                    factor[10] * float(row[census.commute_60_to_89]) +
                    factor[11] * float(row[census.commute_90_or_more]))
        n_samples = mul((1., ) * 12)
        wsum = mul((2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 52.5, 75.0, 120.0))
        return None if n_samples == 0 else wsum / n_samples

    def get_median_household_income(row):
        mhi = float(row[census.median_household_income])
        return mhi

    def get_fraction_owner_occupied(row):
        total = float(row[census.occupied_total])
        owner = float(row[census.occupied_owner])
        return None if total == 0 else owner / total

    d = {}
    # first row has explanations for column names
    labels = census_df.loc[0]
    if False:
        print 'labels'
        for i in xrange(len(labels)):
            print ' ', labels.index[i], labels[i]
    for row_index in xrange(1, len(census_df)):
        if False:
            print row_index
        row = census_df.loc[row_index]  # row is a pd.Series
        if False:
            print row
        ct = get_census_tract(row)
        if ct in d:
            print 'duplicate census tract', ct
            pdb.set_trace()
        ac = get_avg_commute(row)
        mhi = get_median_household_income(row)
        foo = get_fraction_owner_occupied(row)
        if ac is not None and mhi is not None and foo is not None:
            d[ct] = CensusFeatures(avg_commute=ac,
                                   median_hh_income=mhi,
                                   fraction_owner_occupied=foo,
                                   )
    return d


def make_census_reduced_df(d):
    'convert d[census_tract]=(avg commute, med hh inc, fraction owner occ) to dataframe'
    df = pd.DataFrame({'census_tract': [k for k in d.keys()],
                       'avg_commute': [d[k][0] for k in d.keys()],
                       'fraction_owner_occupied': [d[k][2] for k in d.keys()],
                       'median_household_income': [d[k][1] for k in d.keys()]
                       })
    return df


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(base_name=control.arg.base_name)
    print control

    # create dataframes
    deeds_g_al_df = deeds.read_g_al(control.path,
                                    10000 if control.test else None)
    parcels_df = parcels.read(control.path,
                              10000 if control.test else None)
    parcels_sfr_df = parcels_df[parcels.mask_sfr(parcels_df)]

    print 'len deeds g al', len(deeds_g_al_df)
    print 'len parcels', len(parcels_df)
    print 'len parcels sfr', len(parcels_sfr_df)

    # augment parcels and deeds to include a better APN
    print 'adding best apn column for parcels'
    new_column_parcels = best_apn(parcels_sfr_df, parcels.apn_formatted, parcels.apn_unformatted)
    parcels_sfr_df.loc[:, parcels.best_apn] = new_column_parcels  # generates an ignorable warning

    print 'adding best apn column for deeds'
    new_column_deeds = best_apn(deeds_g_al_df, deeds.apn_formatted, deeds.apn_unformatted)
    deeds_g_al_df.loc[:, deeds.best_apn] = new_column_deeds

    # join the files
    print 'starting to merge'
    dp = deeds_g_al_df.merge(parcels_sfr_df, how='inner',
                             left_on=deeds.best_apn, right_on=parcels.best_apn,
                             suffixes=('_deed', '_parcel'))

    print 'names of column in dp dataframe'
    for name in dp.columns:
        print ' ', name

    print 'merge analysis'
    print ' input sizes'

    def ps(name, value):
        s = value.shape
        print '  %20s shape (%d, %d)' % (name, s[0], s[1])

    ps('deeds_g_al_df', deeds_g_al_df)
    ps('parcels_sfr_df', parcels_sfr_df)
    print ' output sizes'
    ps('dp', dp)

    # add in derived parcels features
    parcels_derived_features(parcels_df, dp)  # mutate dp

    # add in census data
    census_df = make_census_reduced_df(reduce_census(read_census(control)))
    dpc = dp.merge(census_df,
                   left_on=parcels.census_tract + '_parcel',
                   right_on="census_tract",
                   )
    print 'len dpc', len(dpc)

    # add in GPS coordinates
    geocoding_df = read_geocoding(control)
    dpcg = dpc.merge(geocoding_df,
                     left_on="best_apn",
                     right_on="G APN",
                     )
    print 'dpcg shape', dpcg.shape

    final = dpcg
    print 'final columns'
    for c in final.columns:
        print c

    print 'final shape', final.shape

    # write merged,augmented dataframe
    print 'writing final dataframe to csv file'
    final.to_csv(control.path_out_transactions)

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

    test_derived()
    main(sys.argv)
