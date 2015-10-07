def columns_contain(s, df):
    'print columns names in df that contain string s'
    'for interactive debugging'
    result = [column_name for column_name in df.columns if s.lower() in column_name.lower()]
    print result
    return result
