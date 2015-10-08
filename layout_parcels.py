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


# encodings for field property indicator, which has codes PROPN
# these codes are MECE and should be used
propn = {
    'single_family_residence': 10,
    'residential_condominium': 11,
    'commercial': 20,
    'duplex': 21,  # duplex, triplex, quadplex
    'apartment': 22,
    'hotel': 23,  # hotel, motel
    'commercial_condominium': 24,
    'retail': 25,
    'service': 26,  # general public
    'office_building': 27,
    'warehouse': 28,
    'financial_institution': 29,
    'medical': 30,  # hospital, medical complex, clinic
    'parking': 31,
    'amusement': 32,  # also: recreation
    'industrial': 50,
    'industrial_light': 51,
    'industrial_heavy': 52,
    'transport': 53,
    'utilities': 54,
    'agriculture': 70,
    'vacant': 80,
    'exempt': 90,
    'not_available': 0,  # also: miscellaneous, none
}


def mask_property_indicator_is(name, df):
    propn_value = propn.get(name, None)
    if propn_value is None:
        print 'bad name passed to mask_property_indicator_is', name
        pdb.set_trace()
    r1 = df[property_indicator] == propn_value
    return r1


def mask_is_sfr(df):
    'NOTE: earlier versions checked land use for value 163'
    return mask_property_indicator_is('single_family_residence', df)


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
