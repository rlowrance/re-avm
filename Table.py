'relational table'

import collections

class Table(object):
    def __init__(self, column_names):
        self._column_names = column_names

    def insert(self, values):
        'values::namedtuple for now'

    def select_where(selected_column, where_columns, filter):
        '''return list with selected_column
        where rows satisify filter(namedtuple) --> bool
        '''
        pass
