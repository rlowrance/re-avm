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
from Path import Path


def usage(msg=None):
    if msg is not None:
        print msg
    print 'usage  : python transactions_subset3.py --input PATH [--test]'
    print ' PATH  : path to input data; ex: ~/Dropbox/real-estate-los-angeles/'
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
        dir_input=pcl.get_arg('--input'),
        test=pcl.has_arg('--test'),
    )
    if arg.dir_input[-1] != '/':
        usage('input PATH must end with /')

    random_seed = 123456
    random.seed(random_seed)

    path = Path(arg.dir_input)

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


def parcels_derived_features(parcels_df):
    'mutate parcels_df by addinging indicator columns for tract and zip5 code features'
    def truncate_zipcode(zip):
        'convert possible zip9 to zip5'
        x = zip / 10000.0 if zip > 99999 else zip
        return int(x if not np.isnan(x) else 0)

    # some zipcodes are 5 digits, others are 9 digits
    # create new feature that has the first 5 digits
    parcels_df['zip5'] = parcels_df[parcels.zipcode].apply(truncate_zipcode)

    def make_items(geo, mask_function, column_name):
        mask = mask_function(parcels_df)
        subset = parcels_df[mask]
        items = subset[column_name]
        r = set(int(item)
                for item in items
                if not np.isnan(item))
        return r

    name_mask = {
        'has_commercial': parcels.mask_commercial,
        'has_industry': parcels.mask_industry,
        'has_park': parcels.mask_park,
        'has_retail': parcels.mask_retail,
        'has_school': parcels.mask_school,
    }

    for geo in ('tract', 'zip5'):
        column_name = parcels.census_tract if geo == 'tract' else parcels.zip5
        for name, mask in name_mask.iteritems():
            new_feature_name = geo + '_' + name
            items = make_items(geo, mask, column_name)
            if len(items) == 0:
                # this happens turning testing and maybe during production
                parcels_df[new_feature_name] = pd.Series(data=False, index=parcels_df.index)
            else:
                for item in make_items(geo, mask, column_name):
                    parcels_df[new_feature_name] = parcels_df[column_name] == item
            print 'new feature %25s is True %6d times' % (
                new_feature_name, sum(parcels_df[new_feature_name]))
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
