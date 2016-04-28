'''define columns in all reports produced by chart06'''
import numpy as np


# all possible column definition
_defs = {
    'median_absolute_error': [6, '%6d', (' ', 'MAE'), 'median absolute error'],
    'model': [5, '%5s', (' ', 'model'),
              'model name (en = elastic net, gd = gradient boosting, rf = random forests)'],
    'n_months_back': [2, '%2d', (' ', 'bk'), 'number of mnths back for training'],
    'max_depth': [4, '%4d', (' ', 'mxd'), 'max depth of any individual decision tree'],
    'n_estimators': [4, '%4d', (' ', 'next'), 'number of estimators (= number of trees)'],
    'max_features': [4, '%4s', (' ', 'mxft'), 'maximum number of features examined to split a node'],
    'learning_rate': [4, '%4.1f', (' ', 'lr'), 'learning rate for gradient boosting'],
    'alpha': [5, '%5.2f', (' ', 'alpha'), 'constant multiplying penalty term for elastic net'],
    'l1_ratio': [4, '%4.2f', (' ', 'l1'), 'l1_ratio mixing L1 and L2 penalties for elastic net'],
    'units_X': [6, '%6s', (' ', 'unitsX'), 'units for the x value; either natural (nat) or log'],
    'units_y': [6, '%6s', (' ', 'unitsY'), 'units for the y value; either natural (nat) or log'],
    'validation_month': [6, '%6s', ('vald', 'month'), 'month used for validation'],
    'rank': [4, '%4d', (' ', 'rank'), 'rank within validation month; 1 == lowest MAE'],
    'median_price': [6, '%6d', ('median', 'price'), 'median price in the validation month'],
    'mae_validation': [6, '%6d', ('vald ', 'MAE'), 'median absolute error in validation month'],
    'mae_next': [6, '%6d', ('next ', 'MAE'),
                 'median absolute error in test month (which follows the validation month)'],
    'note': [15, '%15s', (' ', 'note'),
             'when provided, the next MAE column contains the specified value'],
    'rank_index': [5, '%5d', ('rank', 'index'), 'ranking of model performance in the validation month; 0 == best'],
    'weight': [6, '%6.4f', (' ', 'weight'), 'weight of the model in the ensemble method'],
}


def defs_for_columns(*key_list):
    return [[key] + _defs[key]
            for key in key_list
            ]


def replace_by_spaces(k, v):
    'define values that are replaced by spaces'
    if isinstance(v, float) and np.isnan(v):
        return True
    return False
