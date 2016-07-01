'''create derived features for the year 2000 census

INPUT FILES
 INPUT/corelogic-deeds-*/CAC*.txt

OUTPUT FILES
 WORKING/census-features-derived.csv

The fields in the output csv files are
 index: the code for either the census_tract (6 digits) or zip5 (5 digits)
 geo, same as index
 has_commercial
 has_industry
 has_park
 has_retail
 has_school
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
from Logger import Logger
from Path import Path
from ParseCommandLine import ParseCommandLine


def usage(msg=None):
    if msg is not None:
        print msg
    print 'usage  : python census-features.py [--test]'
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
        path=path,
        path_out=path.dir_working() + arg.base_name + '-' + 'derived.csv',
        random_seed=random_seed,
        test=arg.test,
    )


CensusFeatures = collections.namedtuple(
    'CensusFeatures',
    'avg_commute median_hh_income fraction_owner_occupied',
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
    verbose = False
    labels = census_df.loc[0]
    if verbose and False:
        print 'labels'
        for i in xrange(len(labels)):
            print ' ', labels.index[i], labels[i]
    for row_index in xrange(1, len(census_df)):
        if verbose:
            print row_index
        row = census_df.loc[row_index]  # row is a pd.Series
        if verbose:
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

    # read the census
    census_df = pd.read_csv(
        control.path.dir_input('census'),
        sep='\t',
    )

    derived_df = make_census_reduced_df(reduce_census(census_df))
    derived_df.to_csv(control.path_out)

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
