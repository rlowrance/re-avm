'''define types used by chart06.py and other modules'''

import collections


ModelDescription = collections.namedtuple(
    'ModelDescription',
    'model n_months_back units_X units_y alpha l1_ratio ' +
    'n_estimators max_features max_depth loss learning_rate'
)


ModelResults = collections.namedtuple(
    'ModelResults',
    'rmse mae ci95_low ci95_high predictions'
)
