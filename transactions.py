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
from columns_contain import columns_contain
import layout_deeds as deeds
import layout_parcels as parcels
import layout_transactions as transactions
from Logger import Logger
from ParseCommandLine import ParseCommandLine
from Path import Path
cc = columns_contain


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

    file_out_transactions = arg.base_name + '-al-g-sfr' + ('-test' if arg.test else '') + '.csv'

    return Bunch(
        arg=arg,
        debug=debug,
        max_sale_price=85e6,  # according to Wall Street Journal
        path=path,
        path_in_census_features=path.dir_working() + 'census-features-derived.csv',
        path_in_parcels_features_census_tract=path.dir_working() + 'parcels-features-census_tract.csv',
        path_in_parcels_features_zip5=path.dir_working() + 'parcels-features-zip5.csv',
        path_out_transactions=path.dir_working() + file_out_transactions,
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


def read_census_features(control):
    'return dataframe'
    print 'reading census'
    df = pd.read_csv(control.path_in_census_features, index_col=0)
    return df


def read_geocoding(control):
    'return dataframe'
    print 'reading geocoding'
    df = pd.read_csv(control.path.dir_input('geocoding'), sep='\t')
    return df


def parcels_derived_features(control, transactions_df):
    'return new df by merging df and the geo features'

    # merge in  census tract features
    census_tract_df = pd.read_csv(control.path_in_parcels_features_census_tract, index_col=0)
    m1 = transactions_df.merge(
        census_tract_df,
        how='inner',
        left_on=transactions_df[transactions.census_tract],
        right_on=census_tract_df.geo,
    )
    print 'm1 shape', m1.shape
    pdb.set_trace()  # is there a field has_commercial_census_tract
    cc('commercial', m1)

    # merge in zip5 features
    zip5_df = pd.read_csv(control.path_in_parcels_features_zip5, index_col=0)
    m2 = m1.merge(
        zip5_df,
        how='inner',
        left_on=m1[transactions.zip5],
        right_on=zip5_df.geo,
        suffixes=('_census_tract', '_zip5'),
    )
    print 'm2 shape', m2.shape

    return m2


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(base_name=control.arg.base_name)
    print control

    # NOTE: Organize the computation to minimize memory usage
    # so that this code can run on smaller-memory systems

    def ps(name, value):
        s = value.shape
        print '  %20s shape (%d, %d)' % (name, s[0], s[1])

    # create dataframes
    n_read_if_test = 10000
    deeds_g_al = deeds.read_g_al(
        control.path,
        n_read_if_test if control.test else None,
    )
    parcels_sfr = parcels.read(
        control.path,
        10000 if control.test else None,
        just_sfr=True,
    )
    ps('original deeds g al', deeds_g_al)
    ps('original parcels sfr', parcels_sfr)

    # augment parcels to include a zip5 field (5-digit zip code)
    # drop samples without a zipcode
    # rationale: we use the zip5 to join the features derived from parcels
    # and zip5 is derived from zipcode
    zipcode_present = parcels_sfr[parcels.zipcode].notnull()
    parcels_sfr = parcels_sfr[zipcode_present]
    parcels.add_zip5(parcels_sfr)

    # augment parcels and deeds to include a better APN
    print 'adding best apn column for parcels'
    new_column_parcels = best_apn(parcels_sfr, parcels.apn_formatted, parcels.apn_unformatted)
    parcels_sfr.loc[:, parcels.best_apn] = new_column_parcels  # generates an ignorable warning

    print 'adding best apn column for deeds'
    new_column_deeds = best_apn(deeds_g_al, deeds.apn_formatted, deeds.apn_unformatted)
    deeds_g_al.loc[:, deeds.best_apn] = new_column_deeds

    ps('revised deeds_g_al', deeds_g_al)
    ps('revised parcels_sfr', parcels_sfr)

    # join the deeds and parcels files
    print 'starting to merge'
    m1 = deeds_g_al.merge(parcels_sfr, how='inner',
                          left_on=deeds.best_apn, right_on=parcels.best_apn,
                          suffixes=('_deed', '_parcel'))
    del deeds_g_al
    del parcels_sfr
    ps('m1 merge deed + parcels', m1)

    # add in derived parcels features
    m2 = parcels_derived_features(control, m1)
    ps('ms added parcels_derived', m2)
    del m1

    # add in census data
    census_features_df = read_census_features(control)
    m3 = m2.merge(census_features_df,
                  left_on=transactions.census_tract,
                  right_on="census_tract",
                  )
    del m2
    ps('m3 merged census features', m3)

    # add in GPS coordinates
    geocoding_df = read_geocoding(control)
    m4 = m3.merge(geocoding_df,
                  left_on="best_apn",
                  right_on="G APN",
                  )
    del geocoding_df
    del m3
    ps('m4 merged geocoding', m4)

    final = m4
    print 'final columns'
    for c in final.columns:
        print c,
    print

    cc('fraction', final)  # verify that fraction_owner_occupied is in the output
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
        columns_contain('s', pd.DataFrame())

    main(sys.argv)
