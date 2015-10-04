'hold all the knowledge about the layout and codes of the Corlogic deeds file'
# called record type 1080 in the CoreLogic documentation


import numpy as np
import pandas as pd
import pdb
import zipfile


def is_deeds(df):
    return df.columns[2] == 'MUNICIPALITY CODE'


# map feature names to columns names

apn_formatted = 'APN FORMATTED'
apn_unformatted = 'APN UNFORMATTED'
census_tract = 'CENSUS TRACT'
document_type = 'DOCUMENT TYPE CODE'
has_multiple_apns = 'MULTI APN FLAG CODE'
price = 'SALE AMOUNT'
primary_category = 'PRI CAT CODE'
recording_date = 'RECORDING DATE'
sale_code = 'SALE CODE'
transaction_type = 'TRANSACTION TYPE CODE'


# feature created by our code

best_apn = 'best_apn'


# select certain rows

def is_type(values, types):
    x = values.iloc[0]
    return isinstance(x, types)


def mask_is_arms_length(df):
    values = df[primary_category]
    assert is_type(values, np.str)
    r1 = values == 'A'
    return r1


def mask_is_full_price(df, field_name=sale_code):
    'deed contained the full price'
    values = df[field_name]
    assert is_type(values, str)
    r1 = values == 'F'
    return r1


def mask_is_grant(df):
    values = df[document_type]
    assert is_type(values, np.str)
    r1 = values == 'G'
    return r1


def mask_is_new_construction(df):
    'others include: resale, refinance, ...'
    values = df[transaction_type]
    assert is_type(values, (float, np.int64, np.float64))
    r1 = values == 3.0
    return r1


def mask_is_resale(df):
    'others include: resale, refinance, new contruction, ...'
    values = df[transaction_type]
    assert is_type(values, (float, np.int64, np.float64))
    r1 = values == 1
    return r1


# read file from disk


def read_g_al(path, nrows):
    'return df containing all the grant, arms-length deeds'

    def read_deeds(path_zip_file):
        'return subset df, length of read df'
        z = zipfile.ZipFile(path_zip_file)
        assert len(z.namelist()) == 1
        for archive_member_name in z.namelist():
            print 'opening deeds archive member', archive_member_name
            f = z.open(archive_member_name)
            # line 255719 in one member has an stray " that messes up the csv parser
            skiprows = (255718,) if archive_member_name == 'CAC06037F3.txt' else None
            df = pd.read_csv(f, sep='\t', nrows=nrows,  skiprows=skiprows)
            mask_keep = mask_is_arms_length(df) & mask_is_grant(df)
            keep = df[mask_keep]
            return keep, len(df)

    print 'reading deeds g al'
    dfs = []
    n_read = 0
    for i in (1, 2, 3, 4, 5, 6, 7, 8):
        df, n = read_deeds(path.dir_input('deeds-CAC06037F%d.zip' % i))
        dfs.append(df)
        n_read += n
    all_df = pd.concat(dfs)
    print 'read %d deeds, kept %d, discarded %d' % (n_read, len(all_df), n_read - len(all_df))
    return all_df


if __name__ == '__main__':
    if False:
        pdb.set_trace()
