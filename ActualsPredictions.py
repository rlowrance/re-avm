import datetime
import os
import pandas as pd
import pdb
from pprint import pprint
import unittest

import Date
import layout_transactions
import Path


class ActualsPredictions(object):
    'read and present data from samples2 (actuals) and fit-predict-reduce (predictions)'
    def __init__(self, training_data, test):
        'setup'
        assert training_data in ('train', 'all')
        self.actuals = ActualsPredictions._load_actuals(training_data, test)
        self.predictions = ActualsPredictions._load_predictions(training_data, test)
        self.dates = ActualsPredictions._make_all_dates(self.actuals, self.predictions)

    def actuals_predictions(self, date_apn):
        'return Dict[(fitted, hps_str), (actuals: numpy.ndarray, predictions: numpy.ndarray)]'

    @staticmethod
    def _load_actuals(training_data, test):
        'return Dict[(date: datetime.date, apn: int), price:float]'
        'return Dict[date: datetime.date, Dict[apn, price: float]]'
        path = os.path.join(Path.Path().dir_working(), 'samples2', training_data + '.csv')
        csv = pd.read_csv(
            path,
            nrows=10 if test else None,
            usecols=[
                layout_transactions.sale_date,
                layout_transactions.apn,
                layout_transactions.price,
            ],
            low_memory=False,
        )

        def to_datetime_date(x):
            return Date.Date(x).as_datetime_date()

        # dates = csv[layout_transactions.sale_date].apply(to_datetime_date)
        dates = csv[layout_transactions.sale_date]
        apns = csv[layout_transactions.apn]
        prices = csv[layout_transactions.price]
        result = {}
        for i, date in enumerate(dates):
            dt = Date.Date(date).as_datetime_date()
            result[(dt, apns[i])] = prices[i]
        pdb.set_trace()
        return result

    @staticmethod
    def _load_predictions(training_data, test):
        'return Dict[date: datetime.date, Dict[apn, Dict[fitted, Dict[hps_str, predictions: numpy.ndarray]]]]'

    @staticmethod
    def _make_all_dates(actuals_dict, predictions_dict):
        'return Set[datetime.date]'


class TestActualsPredictions(unittest.TestCase):
    def test_construction(self):
        ap = ActualsPredictions('train', True)
        self.assertTrue(isinstance(ap, ActualsPredictions))


if __name__ == '__main__':
    unittest.main()
    if False:
        pdb
        pprint
