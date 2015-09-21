'hold all the knowledge about the layout and codes of the Corlogic deeds file'
# called record type 1080 in the CoreLogic documentation


import numpy as np
import pandas as pd
import pdb
import sys
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
    print x, type(x)
    return isinstance(x, types)


def mask_arms_length(df):
    values = df[primary_category]
    assert is_type(values, np.str)
    r1 = values == 'A'
    return r1


def mask_full_price(df, field_name=sale_code):
    'deed contained the full price'
    values = df[field_name]
    assert is_type(values, str)
    r1 = values == 'F'
    return r1


def mask_grant(df):
    values = df[document_type]
    assert is_type(values, np.str)
    r1 = values == 'G'
    return r1


def mask_new_construction(df):
    'others include: resale, refinance, ...'
    values = df[transaction_type]
    assert is_type(values, (float, np.int64, np.float64))
    r1 = values == 3.0
    return r1


def mask_resale(df):
    'others include: resale, refinance, new contruction, ...'
    values = df[transaction_type]
    assert is_type(values, (float, np.int64, np.float64))
    r1 = values == 1
    return r1


# read file from disk


def read_g_al(dir_input, nrows):
    'return df containing all the grant, arms-length deeds'

    def read_deed(dir, file_name):
        'return subset df, length of read df'
        z = zipfile.ZipFile(dir + file_name)
        assert len(z.namelist()) == 1
        for archive_member_name in z.namelist():
            print 'opening deeds archive member', archive_member_name
            f = z.open(archive_member_name)
            try:
                # line 255719 has an stray " that messes up the parse
                skiprows = (255718,) if archive_member_name == 'CAC06037F3.txt' else None
                df = pd.read_csv(f, sep='\t', nrows=nrows,  skiprows=skiprows)
                # df = pd.read_csv(f, sep='\t', nrows=10000 if control.test else None)
            except:
                print 'exception', sys.exc_info()[0]
                print 'exception', sys.exc_info()
                pdb.set_trace()
                sys.exit(1)
            mask_keep = mask_arms_length(df) & mask_grant(df)
            keep = df[mask_keep]
            return keep, len(df)

    print 'reading deeds g al'
    dir_deeds_a = dir_input + 'corelogic-deeds-090402_07/'
    df1, l1 = read_deed(dir_deeds_a, 'CAC06037F1.zip')
    df2, l2 = read_deed(dir_deeds_a, 'CAC06037F2.zip')
    df3, l3 = read_deed(dir_deeds_a, 'CAC06037F3.zip')
    df4, l4 = read_deed(dir_deeds_a, 'CAC06037F4.zip')
    dir_deeds_b = dir_input + 'corelogic-deeds-090402_09/'
    df5, l5 = read_deed(dir_deeds_b, 'CAC06037F5.zip')
    df6, l6 = read_deed(dir_deeds_b, 'CAC06037F6.zip')
    df7, l7 = read_deed(dir_deeds_b, 'CAC06037F7.zip')
    df8, l8 = read_deed(dir_deeds_b, 'CAC06037F8.zip')
    df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8])
    lsum = l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8
    print 'read %d deeds, kept %d, discarded %d' % (lsum, len(df), lsum - len(df))
    return df


if __name__ == '__main__':
    if False:
        pdb.set_trace()
