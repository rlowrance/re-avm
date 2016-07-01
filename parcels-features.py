'''create census tract and zip5 features of each parcel

INPUT FILES
 INPUT/corelogic-deeds-*/CAC*.txt

OUTPUT FILES
 WORKING/parcels-features-GEO.csv
 WORKING/parcels-features-GEO-occurs.pickle  how often each feature occurs in the GEO partitioning
 WORKING/parcels-features-zip5.csv

Each parcels was classified as single family retail.

The fields in the output csv files are
 index: the code for either the census_tract (6 digits) or zip5 (5 digits)
 geo, same as index
 has_X where X is in layout_parcels.propn.keys()
'''

import cPickle as pickle
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
        path_out_csv=path.dir_working() + arg.base_name + '-' + arg.geo + '.csv',
        path_out_occurs=path.dir_working() + arg.base_name + '-' + arg.geo + '-occurs.pickle',
        random_seed=random_seed,
        test=arg.test,
    )


def just_used(geo, df):
    'return new DataFrame containing just columns we need for further processing'
    r = pd.DataFrame({
        'geo': df[parcels.census_tract] if geo == 'census_tract' else df[parcels.zip5],
        parcels.land_use: df[parcels.land_use],
        parcels.property_indicator: df[parcels.property_indicator],
    })
    return r


def make_has_indicatorsOLD(df, name_masks):
    'return new df with an index for each geo value'
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
    r['geo'] = r.index
    return r


def make_has_indicators(df, geo_name):
    'return new df with an index for each property indicator value'
    verbose = False
    result_index = set(df.index)
    d = {}       # built up to be the data frame
    occurs = {}  # used for reporting
    format = '%30s occurs in %7d geos'
    for property_indicator_description in parcels.propn.keys():
        feature_name = geo_name + '_has_' + property_indicator_description
        mask = parcels.mask_property_indicator_is(property_indicator_description, df)
        occurs[feature_name] = sum(mask)
        print format % (feature_name, occurs[feature_name])
        is_feature = df[mask]
        d[feature_name] = pd.Series(data=[False] * len(result_index),
                                    index=result_index)
        for is_true in set(is_feature.index):
            if verbose:
                print feature_name, is_true
            d[feature_name][is_true] = True
    total_occurs = reduce(lambda x, y: x + y, occurs.values(), 0)
    print format % ('** any feature **', total_occurs)
    if total_occurs != len(df):
        print total_occurs, len(df)
        pdb.set_trace()  # error detected

    result = pd.DataFrame(data=d)
    result[geo_name] = result.index
    return result, occurs


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

    parcels_df.index = parcels_df.geo  # the index must be the geo field
    n_unique_indices = parcels_df.index.nunique()
    has_indicators, occurs = make_has_indicators(parcels_df, control.arg.geo)

    print 'has_indicators shape', has_indicators.shape
    print '# of unique geo codes', n_unique_indices
    assert has_indicators.shape[0] == n_unique_indices
    if control.test:
        print has_indicators

    # write the results
    has_indicators.to_csv(control.path_out_csv)
    f = open(control.path_out_occurs, 'wb')
    pickle.dump((occurs, control), f)
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
