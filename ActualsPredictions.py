import cPickle as pickle
import os
import pandas as pd
import pdb
from pprint import pprint
import unittest

import layout_transactions
import Path


class ActualsPredictions(object):
    'read and present data from samples2 (actuals) and fit-predict-reduce (predictions)'
    def __init__(self, training_data, test=False, just_200701=False):
        'setup'
        assert training_data in ('train', 'all')
        self.actuals = self._load_actuals(training_data, test)
        self.predictions = self._load_predictions(training_data, just_200701)
        self.transaction_ids = self._make_common_transaction_ids(
            self.actuals,
            self.predictions,
        )
        self.dates = ActualsPredictions._make_all_dates(self.actuals, self.predictions)

    def actuals_predictions(self, date_apn):
        'return Dict[(fitted, hps_str), (actuals: numpy.ndarray, predictions: numpy.ndarray)]'

    def _load_actuals(self, training_data, test):
        'return Dict[(date: datetime.date, apn: int), price:float]'
        'return Dict[date: datetime.date, Dict[apn, price: float]]'
        path = os.path.join(Path.Path().dir_working(), 'samples2', training_data + '.csv')
        if False:
            # I could not get this code to work
            # hence the workaround below
            result = pd.read_csv(
                path,
                usecols=[layout_transactions.price],
                low_memory=False,
                index_col=0,  # the transaction_ids
            )
        else:
            df = pd.read_csv(
                path,
                nrows=10000 if test else None,
                usecols=[
                    layout_transactions.transaction_id,
                    layout_transactions.price,
                ],
                low_memory=False,
            )
            result = pd.DataFrame(
                data={
                    'price': df[layout_transactions.price].values,
                },
                index=df[layout_transactions.transaction_id]
            )
        return result

    def _load_predictions(self, training_data, just_200701):
        'return Dict[date: datetime.date, Dict[apn, Dict[fitted, Dict[hps_str, predictions: numpy.ndarray]]]]'
        filename = 'reduction' + ('_200701' if just_200701 else '') + '.pickle'
        pdb.set_trace()
        path = os.path.join(Path.Path().dir_working(), 'fit-predict-reduce2', filename)
        with open(path, 'r') as f:
            d = pickle.load(f)
        print len(d)
        return d

    def _make_common_transaction_ids(self, actuals, predictions):
        'return Set[TransactionId] that are in both'
        pdb.set_trace()
        transaction_ids_actuals = set(actuals.keys())
        transaction_ids_common = set()
        for fitted, transaction_ids in predictions.iterkeys():
            for prediction_transaction_id in transaction_ids:
                print prediction_transaction_id
                if prediction_transaction_id in transaction_ids_actuals:
                    transaction_ids_common.add(prediction_transaction_id)
        if len(transaction_ids_common) == 0:
            print len(actuals), len(predictions)
            pdb.set_trace()
        return transaction_ids_common


class TestActualsPredictions(unittest.TestCase):
    def test_construction(self):
        ap = ActualsPredictions('train', just_200701=True, test=False)
        self.assertTrue(isinstance(ap, ActualsPredictions))


if __name__ == '__main__':
    unittest.main()
    if False:
        pdb
        pprint
