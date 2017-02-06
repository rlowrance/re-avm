import cPickle as pickle
import os
import pandas as pd
import pdb
from pprint import pprint
import unittest

from Date import Date
import layout_transactions
import Path
from TransactionId2 import TransactionId2


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

        if False:
            # this code does not work because the transaction_ids are read as strings, not TransactionId values
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
        if True:
            # constructing the transaction_ids directly works
            df = pd.read_csv(
                path,
                nrows=10 if test else None,
                usecols=[
                    layout_transactions.sale_date,
                    layout_transactions.apn,
                    layout_transactions.price
                ],
                low_memory=False,
            )
            # make list of transaction_ids
            transaction_ids = []
            for i, sale_date in enumerate(df[layout_transactions.sale_date]):
                transaction_id = TransactionId2(
                    sale_date=Date(from_float=sale_date).as_datetime_date(),
                    apn=long(df[layout_transactions.apn][i]),
                )
                transaction_ids.append(transaction_id)
            result = pd.DataFrame(
                data={
                    'price': df[layout_transactions.price],
                    'transaction_id': transaction_ids},
                index=range(len(transaction_ids)),
            )
        return result

    def _reduction_item_to_s(self, k, v):
        'return str'
        def key_to_str(k):
            fitted, transaction_ids = k
            return 'key(fitted=%s, len(transaction_ids)=%d, transaction_ids[0]=%s)' % (
                    fitted,
                    len(transaction_ids),
                    transaction_ids[0],
            )

        def value_to_str(v):
            assert isinstance(v, dict)
            for k, v in v.iteritems():
                return 'value(first key=%s, first value head=%s)' % (k, v[:5])

        return key_to_str(k) + ' ' + value_to_str(v)

    def _load_predictions(self, training_data, just_200701):
        'return Dict[date: datetime.date, Dict[apn, Dict[fitted, Dict[hps_str, predictions: numpy.ndarray]]]]'
        pdb.set_trace()
        filename = 'reduction' + ('_200701' if just_200701 else '') + '.pickle'
        path = os.path.join(Path.Path().dir_working(), 'fit-predict-reduce2', filename)
        with open(path, 'r') as f:
            d = pickle.load(f)
        assert isinstance(d, dict)
        print 'teduction has %d items' % len(d)
        for k, v in d.iteritems():
            print 'first item in reduction'
            print self._reduction_item_to_s(k, v)
            break
        print 'fitted values from predictions'
        for k in d.iterkeys():
            fitted, transaction_ids = k
            print 'fitted %s %d transaction_ids one sale_date %s' % (
                fitted,
                len(transaction_ids),
                transaction_ids[0].sale_date,
            )
        pdb.set_trace()
        return d

    def _make_common_transaction_ids(self, actuals, predictions):
        'return Set[TransactionId] that are in both'
        pdb.set_trace()
        transaction_ids_actuals = set(actuals.index)
        for transaction_id in transaction_ids_actuals:
            # known date is 2007-01-22
            pdb.set_trace()
            if transaction_id.sale_year.year == 2007 and transaction_id.sale_year.month == 1:
                print transaction_id
        pdb.set_trace()

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
        ap = ActualsPredictions('train', just_200701=True, test=True)
        self.assertTrue(isinstance(ap, ActualsPredictions))


if __name__ == '__main__':
    unittest.main()
    if False:
        pdb
        pprint
