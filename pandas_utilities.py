import pandas as pd
import pdb


def insert(df, new_column_name, new_column_series):
    'mutate df by inserting a new column without a warning'
    df.insert(
        loc=len(df.columns),  # the new column is last
        column=new_column_name,
        values=pd.Series(new_column_series, index=df.index),
    )


def remove(df, column_name):
    'mutate df by removing and return a column'
    df.pop(column_name)


def rename(df, old_column_name, new_column_name):
    'mutate df by renaming iterables of column names'
    # assert len(old_column_names) == len(new_column_names)
    df.rename(
        columns={old_column_name: new_column_name},
        # columns=dict(zip(old_column_names, new_column_names)),
        inplace=True,
    )


if False:
    pdb.set_trace()
