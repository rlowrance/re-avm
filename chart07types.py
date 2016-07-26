'''shared types created by running chart07.py and used by other modules'''

import collections


ReductionKey = collections.namedtuple('ReductionKey', 'test_month')
ReductionValue = collections.namedtuple('ReductionValue', 'model importances mae')
