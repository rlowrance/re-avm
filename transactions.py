'''join deeds and tax roll files to create arms-length, grant, and sfr transactions

INPUT FILES
 INPUT/corelogic-deeds-090402_07/CAC06037F1.zip ...
 INPUT/corelogic-deeds-090402_09/CAC06037F1.zip ...
 INPUT/corelogic-taxrolls-090402_05/CAC06037F1.zip ...
 WORKING/parcels-features-census_tract.csv
 WORKING/parcels-features-zip5.csv

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
       zip5
'''


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
    print 'usage  : python transactions.py [--test]'
    print ' --test: run in test mode'
    sys.exit(1)


def make_control(argv):
    # return a Bunch

    print argv
    if len(argv) not in (1, 2):
        usage('invalid number of arguments')

    pcl = ParseCommandLine(argv)
    if pcl.has_arg('--help'):
        usage()
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
        path_in_census_tract=path.dir_working() + 'parcels-features-census_tract.csv',
        path_in_zip5=path.dir_working() + 'parcels-features-zip5.csv',
        path_out_transactions=path.dir_working() + arg.base_name + '-al-g-sfr.csv',
        random_seed=random_seed,
        test=arg.test,
    )


def features(name, df):
    'return features that contain name'
    return [x for x in df.columns if name.lower() in x.lower()]


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


def parcels_derived_features(control, transactions_df):
    'return new df by merging df and the geo features'

    # merge in  census tract features
    census_tract_df = pd.read_csv(control.path_in_census_tract, index_col=0)
    m1 = transactions_df.merge(
        census_tract_df,
        how='inner',
        left_on=transactions_df[transactions.census_tract],
        right_on=census_tract_df.geo,
        suffixes=(None, '_census_tract'),
    )
    print 'm1 shape', m1.shape

    # merge in zip5 features
    zip5_df = pd.read_csv(control.path_in_zip5, index_col=0)
    m2 = m1.merge(
        zip5_df,
        how='inner',
        left_on=m1[transactions.zip5],
        right_on=zip5_df.geo,
        suffixes=(None, '_zip5'),
    )
    print 'm2 shape', m2.shape

    return m2


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
    parcels_sfr_df = parcels.read(
        control.path,
        10000 if control.test else None,
        just_sfr=True,
    )
    print 'len deeds g al', len(deeds_g_al_df)
    print 'len parcels sfr', len(parcels_sfr_df)

    # augment parcels to include a zip5 field (5-digit zip code)
    # drop samples without a zipcode
    # rationale: we use the zip5 to join the features derived from parcels
    # and zip5 is derived from zipcode
    zipcode_present = parcels_sfr_df[parcels.zipcode].notnull()
    parcels_sfr_df = parcels_sfr_df[zipcode_present]
    parcels.add_zip5(parcels_sfr_df)

    # augment parcels and deeds to include a better APN
    print 'adding best apn column for parcels'
    new_column_parcels = best_apn(parcels_sfr_df, parcels.apn_formatted, parcels.apn_unformatted)
    parcels_sfr_df.loc[:, parcels.best_apn] = new_column_parcels  # generates an ignorable warning

    print 'adding best apn column for deeds'
    new_column_deeds = best_apn(deeds_g_al_df, deeds.apn_formatted, deeds.apn_unformatted)
    deeds_g_al_df.loc[:, deeds.best_apn] = new_column_deeds

    def ps(name, value):
        s = value.shape
        print '  %20s shape (%d, %d)' % (name, s[0], s[1])

    ps('deeds_g_al_df', deeds_g_al_df)
    ps('parcels_sfr_df', parcels_sfr_df)

    # join the deeds and parcels files
    print 'starting to merge'
    dp = deeds_g_al_df.merge(parcels_sfr_df, how='inner',
                             left_on=deeds.best_apn, right_on=parcels.best_apn,
                             suffixes=('_deed', '_parcel'))
    del deeds_g_al_df
    del parcels_sfr_df
    ps('dp', dp)

    print 'names of column in dp dataframe'
    index = 0
    for name in dp.columns:
        print ' ', name,
        index += (index + 1) % 3
    print

    # add in derived parcels features
    dp_parcels_features = parcels_derived_features(control, dp)  # mutate dp
    ps('dp_parcels_features', dp_parcels_features)
    dppf = dp_parcels_features

    # add in census data
    pdb.set_trace()
    pdc = census_derived_features(control, dppf)
    # OLD BELOW ME

    census_df = make_census_reduced_df(reduce_census(read_census(control)))
    dpc = dp.merge(census_df,
                   left_on=parcels.census_tract + '_parcel',
                   right_on="census_tract",
                   )
    print 'len dpc', len(dpc)

    # add in GPS coordinates
    geocoding_df = read_geocoding(control)
    dpcg = dp_parcels_features.merge(geocoding_df,
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

    main(sys.argv)
