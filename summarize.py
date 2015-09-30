import pandas as pd
from pprint import pprint


def summarize(df):
    '''return dataframe summarizing df
    result.index = df.columns
    result.column = attributes of the columns in df
    '''
    description = df.describe()
    print description
    rows = []
    for column_name in df.columns:
        series = df[column_name]
        d = {}
        d['number_nan'] = sum(series.isnull())
        d['number_distinct'] = len(series.unique())
        for index_value in description.index:
            d[index_value] = description[column_name][index_value]
        rows.append(d)
    result = pd.DataFrame(data=rows, index=df.columns)
    return result


if __name__ == '__main__':
    if False:
        pprint()
