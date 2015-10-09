import pandas as pd
import pdb
from pprint import pprint

import columns_contain
cc = columns_contain.columns_contain


def summarize(df):
    '''return dataframe summarizing df
    result.index = df.columns
    result.column = attributes of the columns in df
    '''
    description = df.describe()
    # print description
    print df.shape
    print description.shape
    rows = []
    for column_name in df.columns:
        # print column_name
        if column_name not in description.columns:
            # non-numeric columns are omitted from description
            print 'description is missing', column_name
            continue
        series = df[column_name]
        d = {}
        d['number_nan'] = sum(series.isnull())
        d['number_distinct'] = len(series.unique())
        for statistic_name in description.index:
            d[statistic_name] = description[column_name][statistic_name]
        rows.append(d)
    result = pd.DataFrame(data=rows, index=description.columns)
    return result


if False:
    pprint()
    pdb.set_trace()
