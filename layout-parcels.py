'hold all the knowledge about the layout of the CoreLogic parcels file'
# called record type 2580 in the Corelogic documentation


import numpy as np
import pandas as pd
import pdb
import sys
import zipfile


def is_parcel(df):
    return df.columns[2] == 'APN UNFORMATTED'

# map feature name to fields
assessment_improvement = 'IMPROVEMENT VALUE CALCULATED'
assessment_land = 'LAND VALUE CALCULATED'
assessment_total = 'TOTAL VALUE CALCULATED'
apn_formatted = 'APN FORMATTED'
apn_unformatted = 'APN UNFORMATTED'
census_tract = 'CENSUS TRACT'
effective_year_built = 'EFFECTIVE YEAR BUILT'
n_buildings = 'NUMBER OF BUILDINGS'
n_rooms = 'TOTAL ROOMS'
n_units = 'UNITS NUMBER'
property_indicator = 'PROPERTY INDICATOR CODE'
land_size = 'LAND SQUARE FOOTAGE'
land_use = 'UNIVERSAL LAND USE CODE'
living_size = 'LIVING SQUARE FEET'
year_built = 'YEAR BUILT'
zipcode = 'PROPERTY ZIPCODE'  # not owner zipcode


# feature created by our code
best_apn = 'best_apn'


# select rows with certain properties


def mask_commercial(df):
    values = df[property_indicator]
    assert isinstance(values.iloc[0], np.int64)
    r1 = values == 30  # commercial
    r2 = values == 24  # commercial (condominium)
    return r1 | r2


def mask_industry(df):
    values = df[property_indicator]
    assert isinstance(values.iloc[0], np.int64)
    r1 = values == 50  # industry
    r2 = values == 51  # industry light
    r3 = values == 52  # industry heavy
    return r1 | r2 | r3


def mask_park(df):
    values = df[land_use]
    assert isinstance(values.iloc[0], np.int64)
    r1 = values == 757  # park
    return r1


def mask_sfr(df):
    values = df[land_use]
    assert isinstance(values.iloc[0], np.int64)
    r1 = values == 163  # single family residential
    return r1


def mask_retail(df):
    values = df[property_indicator]
    assert isinstance(values.iloc[0], np.int64)
    r1 = values == 25  # retail
    return r1


def mask_school(df):
    values = df[land_use]
    assert isinstance(values.iloc[0], np.int64)
    r1 = values == 650  # school
    r2 = values == 652  # nursery school
    r3 = values == 654  # high school
    r4 = values == 655  # private school
    # not included:
    #  656 vocational/trade school
    #  660 education service
    #  680 university
    r5 = values == 664  # secondary educational school
    r6 = values == 665  # public school
    return r1 | r2 | r3 | r4 | r5 | r6


def read_driver(dir_input, nrows, just_sfr):
    'return df containing all parcels (not just single family residences)'
    def read_parcels(dir, file_name):
        'return subset kept (which is all), length of read df'

        def make_sfr(df):
            keep = df[mask_sfr(df)]
            return keep, len(keep)

        z = zipfile.ZipFile(dir + file_name)
        assert len(z.namelist()) == 1
        for archive_member_name in z.namelist():
            f = z.open(archive_member_name)
            try:
                df = pd.read_csv(f, sep='\t', nrows=nrows)
            except:
                print 'exception', sys.exc_info()[0]
            return make_sfr(df) if just_sfr else df, len(df)

    print 'reading parcels'
    dir_parcels = dir_input + 'corelogic-taxrolls-090402_05/'
    df1, l1 = read_parcels(dir_parcels, 'CAC06037F1.zip')
    df2, l2 = read_parcels(dir_parcels, 'CAC06037F2.zip')
    df3, l3 = read_parcels(dir_parcels, 'CAC06037F3.zip')
    df4, l4 = read_parcels(dir_parcels, 'CAC06037F4.zip')
    df5, l5 = read_parcels(dir_parcels, 'CAC06037F5.zip')
    df6, l6 = read_parcels(dir_parcels, 'CAC06037F6.zip')
    df7, l7 = read_parcels(dir_parcels, 'CAC06037F7.zip')
    df8, l8 = read_parcels(dir_parcels, 'CAC06037F8.zip')
    df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8])
    lsum = l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8
    print 'read %d parcels, kept %d, discarded %d' % (lsum, len(df), lsum - len(df))
    return df


def read(dir_input, nrows):
    return read_driver(just_sfr=False, dir_input=dir_input, nrows=nrows)


def read_sfr(dir_input, nrows):
    return read_driver(just_sfr=True, dir_input=dir_input, nrows=nrows)


if __name__ == '__main__':
    if False:
        pdb.set_trace()
