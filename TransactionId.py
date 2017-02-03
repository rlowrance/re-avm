import collections
import datetime
import unittest

from Date import Date


TransactionId = collections.namedtuple('TransactionId', 'sale_date apn')


def canonical(transaction_id):
    'return new in standard format'
    def canonical_sale_date(date):
        'return a datetime.date'
        if isinstance(date, datetime.date):
            return date
        elif isinstance(date, float):
            return Date(from_float=date).as_datetime_date()
        else:
            raise ValueError('unsupported date type: %s' % date)

    def canonical_apn(apn):
        'return an long'
        value = long(apn)
        if value != apn:
            raise ValueError('loss of precision: %s' % apn)
        return value

    return TransactionId(
        sale_date=canonical_sale_date(transaction_id.sale_date),
        apn=canonical_apn(transaction_id.apn)
    )


class TestTransactionId(unittest.TestCase):
    def test_equality(self):
        'equality is based on field value, not object identity'
        x = TransactionId(1, 23)
        y = TransactionId(1, 23)
        z = TransactionId(2, 23)
        self.assertEqual(x, y)
        self.assertNotEqual(x, z)

    def test_canonical(self):
        x = TransactionId(sale_date=20070124.0, apn=2425019009L)
        y = canonical(x)
        self.assertTrue(isinstance(y.sale_date, datetime.date))
        self.assertTrue(isinstance(y.apn, long))
        self.assertNotEqual(x, y)


if __name__ == '__main__':
    unittest.main()
