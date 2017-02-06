import datetime
import pdb
import unittest


class TransactionId2(object):
    def __init__(self, sale_date=None, apn=None):
        assert sale_date is not None, sale_date
        assert apn is not None, apn
        assert isinstance(sale_date, datetime.date), sale_date
        assert isinstance(apn, long), apn
        self.sale_date = sale_date
        self.apn = apn

    def __str__(self):
        'return informal string representation'
        return '%sAPN%d' % (self.sale_date, self.apn)

    def __repr__(self):
        'return official string representation'
        return 'TransactionId2(sale_date=%s,apn=%sL)' % (self.sale_date, self.apn)

    def __eq__(self, other):
        return self.sale_date == other.sale_date and self.apn == other.apn

    def __lt__(self, other):
        if self.sale_date == other.sale_date:
            return self.apn < other.apn
        else:
            return self.sale_date < other.sale_date

    def __hash__(self):
        return hash((self.sale_date, self.apn))


class TestTransactionId2(unittest.TestCase):
    def test_construction_ok(self):
        for test in (
            [2001, 2, 3, 10],
        ):
            year, month, day, apn = test
            date_value = datetime.date(year, month, day)
            apn_value = long(apn)
            x = TransactionId2(sale_date=date_value, apn=apn_value)
            self.assertTrue(isinstance(x, TransactionId2))
            self.assertEqual(date_value, x.sale_date)
            self.assertEqual(apn_value, x.apn)

    def test_construction_bad(self):
        def make(year, month, day, apn):
            return TransactionId2(sale_date=year * 10000 + month * 100 + day, apn=apn)

        for test in (
            [2001, 2, 3, 10],
        ):
            year, month, day, apn = test
            self.assertRaises(AssertionError, make, year, month, day, apn)

    def test_str_repr(self):
        verbose = True
        for test in (
            [2001, 2, 3, 10],
        ):
            year, month, day, apn = test
            date_value = datetime.date(year, month, day)
            apn_value = long(apn)
            x = TransactionId2(sale_date=date_value, apn=apn_value)
            if verbose:
                print x  # calls __str__
                print x.__repr__()

    def test_eq(self):
        a1 = TransactionId2(sale_date=datetime.date(2001, 2, 3), apn=10L)
        a2 = TransactionId2(sale_date=datetime.date(2001, 2, 3), apn=10L)
        b = TransactionId2(sale_date=datetime.date(2001, 2, 4), apn=10L)
        c = TransactionId2(sale_date=datetime.date(2001, 2, 3), apn=11L)
        self.assertEqual(a1, a2)
        self.assertNotEqual(a1, b)
        self.assertNotEqual(a1, c)

    def test_lt(self):
        a = TransactionId2(sale_date=datetime.date(2001, 2, 3), apn=10L)
        b = TransactionId2(sale_date=datetime.date(2001, 2, 3), apn=11L)
        c = TransactionId2(sale_date=datetime.date(2001, 2, 5), apn=10L)
        self.assertLess(a, b)
        self.assertLess(a, c)
        self.assertLess(b, c)

    def test_has(self):
        'test by making a set'
        a = TransactionId2(sale_date=datetime.date(2001, 2, 3), apn=10L)
        b = TransactionId2(sale_date=datetime.date(2001, 2, 3), apn=11L)
        c = TransactionId2(sale_date=datetime.date(2001, 2, 5), apn=10L)
        x = set((a, b, c))
        self.assertEqual(3, len(x))


if __name__ == '__main__':
    unittest.main()
    if False:
        pdb
