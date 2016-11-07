'''define types used by chart06.py and other modules'''

import collections
import numpy as np


ModelDescription = collections.namedtuple(
    'ModelDescription',
    'model n_months_back units_X units_y alpha l1_ratio ' +
    'n_estimators max_features max_depth loss learning_rate'
)


ModelResults = collections.namedtuple(
    'ModelResults',
    'rmse mae ci95_low ci95_high predictions'
)


class ColumnDefinitions(object):
    'all reports use these column definitions'
    def __init__(self):
        self._defs = {
            'alpha': [5, '%5.2f', (' ', 'alpha'), 'constant multiplying penalty term for elastic net'],
            'description': [50, '%50s', ('model', 'description'),
                            'for experts, model_name(n_month_back, n_estimators, max_depth[, learning_rate])'],
            'fraction_median_price_next_month_best': [
                6, '%6.3f', ('best', 'relerr'),
                'best model MAE as a fraction of the median price in the next month'],
            'fraction_median_price_next_month_ensemble': [
                6, '%6.3f', ('ensmbl', 'relerr'),
                'ensemble MAE as a fraction of the median price in the next month'],
            'fraction_median_price_next_month_index0': [
                6, '%6.3f', ('rank1', 'relerr'),
                'rank 1 MAE as a fraction of the median price in the next month'],
            'k': [3, '%3d', (' ', 'K'), 'number of experts'],
            'learning_rate': [4, '%4.1f', (' ', 'lr'), 'learning rate for gradient boosting'],
            'l1_ratio': [4, '%4.2f', (' ', 'l1'), 'l1_ratio mixing L1 and L2 penalties for elastic net'],
            'mae_best_next_month': [6, '%6d', ('best', 'MAE'),
                                    'median absolute error of the best model in the next month'],
            'mae_ensemble': [6, '%6d', ('ensb', 'MAE'),
                             'median absolute error of ensemble model'],
            'mae_index0': [6, '%6d', ('rank1', 'MAE'), 'median absolute error of rank 1 model'],
            'mae_next': [6, '%6d', ('next', 'MAE'),
                         'median absolute error in test month (which follows the validation month)'],
            'mae_query': [6, '%6d', ('query', 'MAE'),
                          'median absolute error in query month (which follows the validation month)'],
            'mae_validation': [6, '%6d', ('vald', 'MAE'), 'median absolute error in validation month'],
            'mare_query': [7, '%7.4f', ('query', 'MARE'),
                           'median absolute errors as percent of median price in query month'],
            'mare_validation': [7, '%7.4f', ('vald', 'MARE'),
                                'median absolute error as a percent of the median price in the validation month'],
            'max_depth': [4, '%4d', (' ', 'mxd'), 'max depth of any individual decision tree'],
            'max_features': [4, '%4s', (' ', 'mxft'), 'maximum number of features examined to split a node'],
            'median_absolute_error': [6, '%6d', (' ', 'MAE'), 'median absolute error'],
            'median_price': [6, '%6d', ('median', 'price'), 'median price in the validation month'],
            'model': [5, '%5s', (' ', 'model'),
                      'model name (en = elastic net, gd = gradient boosting, rf = random forests)'],
            'n_months_back': [2, '%2d', (' ', 'bk'), 'number of mnths back for training'],
            'n_estimators': [4, '%4d', (' ', 'nest'), 'number of estimators (= number of trees)'],
            'oracle_less_best': [10, '%10.0f', ('oracle', '- best'),
                                 'mae of oracle - mae of best model in validation period'],
            'oracle_less_ensemble': [10, '%10.0f', ('oracle', '- ensemble'),
                                     'mae of oracle - mae of ensemble model in validation period'],
            'rank': [4, '%4d', (' ', 'rank'), 'rank within validation month; 1 == lowest MAE'],
            'rank_index': [5, '%5d', ('rank', 'index'),
                           'ranking of model performance in the validation month; 0 == best'],
            'units_X': [6, '%6s', (' ', 'unitsX'), 'units for the x value; either natural (nat) or log'],
            'units_y': [6, '%6s', (' ', 'unitsY'), 'units for the y value; either natural (nat) or log'],
            'validation_month': [6, '%6s', ('vald', 'month'), 'month used for validation'],
            'weight': [6, '%6.4f', (' ', 'weight'), 'weight of the model in the ensemble method'],
        }

    def defs_for_columns(self, *key_list):
        return [[key] + self._defs[key]
                for key in key_list
                ]

    def replace_by_spaces(self, k, v):
        'define values that are replaced by spaces'
        if isinstance(v, float) and np.isnan(v):
            return True
        return False
