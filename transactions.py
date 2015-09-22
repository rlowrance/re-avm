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
'''

import collections
import numpy as np
import pandas as pd
import cPickle as pickle
import pdb
from pprint import pprint
import random
import sys
import time


from Bunch import Bunch
from directory import directory
import layout_census as census
import layout_deeds as deeds
import layout_parcels as parcels
from Logger import Logger
from ParseCommandLine import ParseCommandLine


def usage(msg=None):
    if msg is not None:
        print msg
    print 'usage  : python transactions_subset3.py [--just TAG] [--test]'
    print ' TAG   : run only a portion of the analysis (used during development)'
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

    debug = False

    return Bunch(
        arg=arg,
        debug=debug,
        test=arg.test,
        path_in_census=directory('input') + 'neighborhood-data/census.csv',
        path_in_geocoding=directory('input') + 'geocoding.tsv',
        path_out_transactions=directory('working') + arg.base_name + '-al-g-sfr.csv',
        dir_deeds_a=directory('input') + 'corelogic-deeds-090402_07/',
        dir_deeds_b=directory('input') + 'corelogic-deeds-090402_09/',
        dir_parcels=directory('input') + 'corelogic-taxrolls-090402_05/',
        max_sale_price=85e6,  # according to Wall Street Journal
        random_seed=random_seed,
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
    df = pd.read_csv(control.path_in_census, sep='\t')
    return df


def read_geocoding(control):
    'return dataframe'
    print 'reading geocoding'
    df = pd.read_csv(control.path_in_geocoding, sep='\t')
    return df


def parcels_derived_features(parcels_df):
    'mutate parcels_df by addinging indicator columns for tract and zip code features'
    def truncate_zipcode(zip):
        'convert possible zip9 to zip5'
        x = zip / 10000.0 if zip > 99999 else zip
        return int(x if not np.isnan(x) else 0)

    def make_tracts(mask_function):
        'return set of tracts containing the specified feature'
        mask = mask_function(parcels_df)
        subset = parcels_df[mask]
        r = set(int(item)
                for item in subset[parcels.census_tract]
                if not np.isnan(item))
        return r

    def make_zips(mask_function):
        'return set of tracts containing the specified feature'
        mask = mask_function(parcels_df)
        subset = parcels_df[mask]
        r = set(truncate_zipcode(item)
                for item in subset[parcels.zipcode]
                if not np.isnan(item))
        return r

    # add columns for censu tract indicators
    tracts = {
        'has_commercial': make_tracts(parcels.mask_commercial),
        'has_industry': make_tracts(parcels.mask_industry),
        'has_park': make_tracts(parcels.mask_park),
        'has_retail': make_tracts(parcels.mask_retail),
        'has_school': make_tracts(parcels.mask_school),
    }
    for k, tract_set in tracts.iteritems():
        for tract in tract_set:
            parcels_df['tract ' + k] = parcels_df[parcels.census_tract] == tract

    # add columns for zip code indicators
    zips = {
        'has_commercial': make_zips(parcels.mask_commercial),
        'has_industry': make_zips(parcels.mask_industry),
        'has_park': make_zips(parcels.mask_park),
        'has_retail': make_zips(parcels.mask_retail),
        'has_school': make_zips(parcels.mask_school),
    }
    truncated_zipcodes = parcels_df[parcels.zipcode].apply(truncate_zipcode)
    for k, zip_set in zips.iteritems():
        for zip5 in zip_set:
            parcels_df['zip ' + k] = truncated_zipcodes == zip5

    # report on geographic features
    for geo in ('tract', 'zip'):
        for feature in ('has_commercial', 'has_industry', 'has_park', 'has_retail', 'has_school'):
            print 'location kind %6s %15s count: %7d' % (
                geo, feature, sum(parcels_df[geo + ' ' + feature])
            )

    return


def just_timing(control):
    '''report timing for long-IO operations

    SAMPLE OUTPUT
    read all parcel files : 103 sec
    dump parcels to pickle: 338
    read pickle           : 760
    write parcels to csv  :  87
    read csv engine python: 218
    read csv engine c     :  38
    '''
    if False:
        start = time.time()
        parcels.read(directory('input'), control.test)
        print 'read parcels:', time.time() - start

    path_csv = '/tmp/parcels.csv'
    path_pickle = '/tmp/parcels.pickle'

    if False:
        start = time.time()
        f = open(path_pickle, 'wb')
        pickle.dump(parcels, f)
        f.close()
        print 'dump parcels in pickle form:', time.time() - start

    if False:
        start = time.time()
        f = open(path_pickle, 'rb')
        pickle.load(f)
        f.close()
        print 'load parcels from pickle file:', time.time() - start

    if False:
        start = time.time()
        parcels.to_csv(path_csv)
        print 'write parcels to csv:', time.time() - start

    start = time.time()
    pd.read_csv(path_csv, engine='python')
    print 'read parcels from csv file, parser=python:', time.time() - start

    start = time.time()
    pd.read_csv(path_csv, engine='c')
    print 'read parcels from csv file, parser=c:', time.time() - start


def read_and_write(read_function, write_path, control):
    start = time.time()
    df = read_function(directory('input'), control.test)
    print 'secs to read:', time.time() - start
    start = time.time()
    df.to_csv(write_path)
    print 'secs to write:', time.time() - start


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


def just_cache(control):
    'consolidate the parcels and deeds files into 2 csv files'
    # goal: speed up testing but don't use in production

    print 'deeds g al'
    read_and_write(deeds.read_g_al, control.path_cache_base + 'deeds-g-al.csv', control)

    print 'parcels'
    read_and_write(parcels.read, control.path_cache_base + 'parcels.csv', control)


def just_parcels(control):
    print 'parcels'
    read_and_write(parcels.read, control.path_cache_base + 'parcels.csv', control)


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(base_name=control.arg.base_name)
    print control

    if control.just:
        if control.just == 'timing':
            just_timing(control)
        elif control.just == 'cache':
            just_cache(control)
        elif control.just == 'parcels':
            just_parcels(control)
        else:
            assert False, control.just
        pdb.set_trace()
        print 'DISCARD RESULTS; JUST', control.just
        sys.exit(1)

    # create dataframes
    deeds_g_al_df = deeds.read_g_al(directory('input'),
                                    10000 if control.test else None)
    parcels_df = parcels.read(directory('input'),
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
    parcels_derived_features(parcels_df)  # mutate parcels_df

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
