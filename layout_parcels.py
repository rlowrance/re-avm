'hold all the knowledge about the layout of the CoreLogic parcels file'
# called record type 2580 in the Corelogic documentation


import numpy as np
import pandas as pd
import pdb
import zipfile


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
zip5 = 'zip5'
zip9 = zipcode


def dataframe_is_parcel(df):
    return df.columns[2] == 'APN UNFORMATTED'


# select rows with certain properties

def mask_parcel_has_census_tract(df):
    values = df[census_tract]
    r = values.notnull()
    return r


def mask_parcel_has_zipcode(df):
    values = df[zipcode]
    r = values.notnull()
    return r


def mask_is_commercial(df):
    values = df[property_indicator]
    r1 = values == 30  # commercial
    r2 = values == 24  # commercial (condominium)
    return r1 | r2


def mask_is_industry(df):
    values = df[property_indicator]
    r1 = values == 50  # industry
    r2 = values == 51  # industry light
    r3 = values == 52  # industry heavy
    return r1 | r2 | r3


def mask_is_park(df):
    values = df[land_use]
    r1 = values == 757  # park
    return r1


def mask_is_sfr(df):
    values = df[land_use]
    r1 = values == 163  # single family residential
    return r1


def mask_is_retail(df):
    values = df[property_indicator]
    r1 = values == 25  # retail
    return r1


def mask_is_school(df):
    values = df[land_use]
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


def add_zip5(df):
    '''mutate df by appending column with zip5 values

    NOTE: some of the zipcode values are already 5 digit values
    '''

    def to_int(x):
        return x if np.isnan(x) else int(x)

    zip9_values = df[zipcode]
    zip5_values = (zip9_values / 10000.0).apply(to_int)
    values = np.where(zip9_values <= 99999, zip9_values, zip5_values)
    df[zip5] = pd.Series(data=values,
                         dtype=np.int32,
                         )
    return
    df[zip5] = pd.Series(data=(df[zipcode] / 10000.0).apply(to_int),
                         dtype=np.int32,
                         index=df.index)


def test_add_zip5():
    verbose = False
    df = pd.DataFrame({zipcode: (123456790.0, 98765.0, np.nan)})
    add_zip5(df)
    if verbose:
        print df.columns
        print df


test_add_zip5()


def read(path, nrows, just_sfr=False):
    'return df containing all parcels or just the single-family residential parcels'

    def read_parcels(path_zip_file):
        'return subset kept (which is all), length of read df'

        z = zipfile.ZipFile(path_zip_file)
        assert len(z.namelist()) == 1
        for archive_member_name in z.namelist():
            print 'opening parcels archive member', archive_member_name
            f = z.open(archive_member_name)
            try:
                df = pd.read_csv(f, sep='\t', nrows=nrows)
            except Exception as e:
                print 'exception reading archive member ', archive_member_name
                print 'the exception', e
                pdb.set_trace()
            return (
                df[mask_is_sfr(df)] if just_sfr else df,
                len(df),
            )

    print 'reading parcels'
    dfs = []
    n_read = 0
    for i in (1, 2, 3, 4, 5, 6, 7, 8):
        df, n = read_parcels(path.dir_input('parcels-CAC06037F%d.zip' % i))
        dfs.append(df)
        n_read += n
    all_df = pd.concat(dfs)
    print 'read %d parcels, kept %d, discarded %d' % (n_read, len(all_df), n_read - len(all_df))
    return all_df


if __name__ == '__main__':
    if False:
        pdb.set_trace()
