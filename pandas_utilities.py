import pandas as pd
import pdb


def df_append_column(df, new_column_name, new_column_series):  # DEPRECATED
    'mutate df by inserting a new column without a warning'
    df.insert(
        loc=len(df.columns),  # the new column is last
        column=new_column_name,
        values=pd.Series(new_column_series, index=df.index),
    )


def df_iterate_over_rows(df):
    'iterate over index items and rows (as pd.Series)'
    # NOTE: or just call df.iterrows() directly
    for index, row in df.iterrows():
        yield index, row


def df_remove_column(df, column_name):
    'mutate df by removing and returning a column'
    df.pop(column_name)


def df_rename_column(df, old_column_name, new_column_name):
    'mutate df by renaming iterables of column names'
    # assert len(old_column_names) == len(new_column_names)
    df.rename(
        columns={old_column_name: new_column_name},
        # columns=dict(zip(old_column_names, new_column_names)),
        inplace=True,
    )


if False:
    pdb.set_trace()
