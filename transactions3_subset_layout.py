'hold all the knowledge about the layout and codes of the transactions3 file'


import datetime
import numpy as np
import pdb


# map feature names to column names
apn = 'best_apn'
assessment_improvement = 'ASSD IMPROVEMENT VALUE'
assessment_land = 'ASSD LAND VALUE'
assessment_total = 'ASSD TOTAL VALUE'
census_tract = 'CENSUS TRACT_parcel'
gps_latitude = 'G LATITUDE'
gps_longitude = 'G LONGITUDE'
land_size = 'LAND SQUARE FOOTAGE'
living_size = 'LIVING SQUARE FEET'
municipality_name = 'MUNICIPALITY NAME'
multi_apn_flag_code = 'MULTI APN FLAG CODE_deed'
n_buildings = 'NUMBER OF BUILDINGS'
n_rooms = 'TOTAL ROOMS'
n_units = 'UNITS NUMBER'
price = 'SALE AMOUNT_deed'
recording_date = 'RECORDING DATE_deed'
rooms = 'TOTAL ROOMS'
sale_code = 'SALE CODE_deed'
sale_date = 'SALE DATE_deed'
township = 'TOWNSHIP'
transaction_type_code = 'TRANSACTION TYPE CODE'
year_built = 'YEAR BUILT'
year_built_effective = 'EFFECTIVE YEAR BUILT'
zoning = 'ZONING'

# fields created in transactions3-subset.py
sale_date_python = 'sale_date_python'
yyyymm = 'yyyymm'


# masks for selecting fields with certain coded values
# each returns a boolean series or a list of field names used

def mask_full_price(df):
    'deed has full price; decode SCODE'
    value = df[sale_code]
    r = value == 'F'  # SCODE: sale price full
    # Other code are for sale price partial, lease, not of public record, ...
    return r


def mask_gps_latitude_known(df):
    value = df[gps_latitude]
    if np.isnan(value).any():
        print 'contains NaN'
        pdb.set_trace()
    r = value != 0
    return r


def mask_gps_longitude_known(df):
    value = df[gps_longitude]
    if np.isnan(value).any():
        print 'contains NaN'
        pdb.set_trace()
    r = value != 0
    return r


def mask_is_one_building(df):
    value = df[n_buildings]
    r = value == 1
    return r


def mask_is_one_parcel(df):
    def one_parcel(x):
        'decode record type 1080 field SLMLT'
        # the presence of a code indicates more than one parcel was involved
        return isinstance(x, float) and np.isnan(x)

    value = df[multi_apn_flag_code]  # record type 1080 field SLMLT
    r = value.apply(one_parcel)
    return r


def mask_new_or_resale(df):
    'decode transaction type to detect new sales and resales'
    def new_or_resale(x):
        return (
            x == 1 or  # resale
            x == 3     # new construction or subdivision
        )

    value = df[transaction_type_code]  # TRNTP
    r = value.apply(new_or_resale)
    return r


def mask_sale_date_present(df):
    def present(x):
        return not np.isnan(x)

    value = df[sale_date]
    r = value.apply(present)
    return r


def mask_sale_date_valid(df):
    def ymd(date):
        'yyyymmdd --> date(year, month, day)'
        year = int(date / 10000)
        md = date - year * 10000
        month = int(md / 100)
        day = md - month * 100
        try:
            datetime.date(int(year), int(month), int(day))
            return True
        except:
            return False

    value = df[sale_date]
    r = value.apply(ymd)
    return r


def mask_sold_after_2002(df):
    def after_2002(x):
        if np.isnan(x):
            return False
        else:
            return int(x / 10000) > 2002  # year > 2002

    values = df[sale_date]
    r = values.apply(after_2002)
    return r


# specialized I/O

if False:
    pdb.set_trace()
