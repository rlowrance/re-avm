'A month, as in 200703; immutable'

import datetime
import pdb
import unittest


class Month(object):
    def __init__(self, value1, value2=None):
        '''Constructors
           Month('200703')
           Month(200703)
           Month(2007, 3)
           Month(datetime.date)
        '''

        if value2 is None:
            if isinstance(value1, str):
                self.year = int(value1[:4])
                self.month = int(value1[4:])
            elif isinstance(value1, int):
                self.month = value1 % 100
                self.year = (value1 - self.month) / 100
            elif isinstance(value1, Month):
                self.month = value1.month
                self.year = value1.year
            elif isinstance(value1, datetime.date):
                self.month = value1.month
                self.year = value1.year
            else:
                print 'construction error: value1 is of type %s' % type(value1), value1
        else:
            self.year = int(value1)
            self.month = int(value2)

        # enforce invariant (other methods depend on this)
        assert self.year > 0, self
        assert 1 <= self.month <= 12, self

    def __repr__(self):
        return 'Month(year=%d, month=%d)' % (self.year, self.month)

    def increment(self, by=1):
        'return new Month one month after self'
        assert by >= 0, by
        month = self.month + by
        if month > 12:
            delta_years = month // 12
            month = month - 12 * delta_years
            year = self.year + delta_years
            return Month(year, month)
        else:
            return Month(self.year, month)

    def decrement(self, by=1):
        'return new Month one month before self'
        assert by >= 0, by
        month = self.month - by
        year = self.year
        while month <= 0:
            month += 12
            year -= 1
        return Month(year, month)

    def as_str(self):
        return '%04d%02d' % (self.year, self.month)

    def as_int(self):
        return self.year * 100 + self.month

    def equal(self, other):
        return self.year == other.year and self.month == other.month

    def __eq__(self, other):
        result = self.year == other.year and self.month == other.month
        return result

    def __hash__(self):
        return hash((self.year, self.month))  # hash of tuple of properties


class TestMonth(unittest.TestCase):
    def test_from_datetimedate(self):
        pdb.set_trace()
        dt = datetime.date(2007, 5, 2)
        m = Month(dt)
        self.assertTrue(m.year == 2007)
        self.assertTrue(m.month == 5)
        
        
    def test_eq_2_with_same_content(self):
        a = Month(2003, 1)
        b = Month(2003, 1)
        self.assertTrue(a == b)

    def test_eq_copy(self):
        a = Month(2003, 1)
        b = a
        self.assertTrue(a == b)

    def test_neq_different(self):
        a = Month(2003, 1)
        b = Month(2003, 2)
        self.assertFalse(a == b)
        self.assertFalse(b == a)
        c = Month(2004, 1)
        self.assertFalse(a == c)
        self.assertFalse(c == a)

    def test_eq_same(self):
        a = Month(2003, 1)
        self.assertTrue(a == a)

    def test_set_with_same_element_len_2b(self):
        a = Month(2003, 1)
        b = Month(2003, 2)
        c = Month(2003, 1)
        assert a == c, (a, c)
        s = set([a, b, c])
        assert len(s) == 2
        self.assertEqual(len(s), 2)
        return

    def test_set_with_2_element_len_2(self):
        a = Month(2003, 1)
        b = Month(2003, 2)
        s = set([a, b])
        self.assertEqual(len(s), 2)
        return

    def test_set_with_same_element_len_1b(self):
        a = Month(2003, 1)
        a2 = a
        s = set([a, a2])
        self.assertEqual(len(s), 1)
        return

    def test_set_with_same_element_len_1(self):
        a = Month(2003, 1)
        s = set([a, a])
        self.assertEqual(len(s), 1)
        return

    def test_eq_based_on_content(self):
        a = Month(2003, 1)
        b = Month(2003, 1)
        c = Month(2003, 2)
        self.assertTrue(a == b)
        self.assertFalse(a == c)
        self.assertTrue(a == a)

    def test_constructor(self):
        self.assertTrue(Month('200703').equal(Month(2007, 03)))
        self.assertTrue(Month(200703).equal(Month(2007, 03)))
        self.assertTrue(Month(200712).equal(Month(2007, 12)))
        m1 = Month(2007, 3)
        m2 = Month(m1)
        self.assertTrue(m1 != m2)

    def test_as_str(self):
        self.assertTrue(Month(200703).as_str() == '200703')

    def test_as_int(self):
        self.assertTrue(Month(200703).as_int() == 200703)

    def test_equal(self):
        self.assertTrue(Month('200703').equal(Month(2007, 03)))

    def test_increment(self):
        self.assertTrue(Month(200612).increment().equal(Month(200701)))
        self.assertTrue(Month(200612).increment(1).equal(Month(200701)))
        self.assertTrue(Month(200612).increment(2).equal(Month(200702)))
        self.assertTrue(Month(200612).increment(14).equal(Month(200802)))
        self.assertTrue(Month(200701).increment().equal(Month(200702)))
        self.assertTrue(Month(200712).increment().equal(Month(200801)))

    def test_decrement(self):
        self.assertTrue(Month(200701).decrement().equal(Month(200612)))
        self.assertTrue(Month(200701).decrement(1).equal(Month(200612)))
        self.assertTrue(Month(200701).decrement(2).equal(Month(200611)))
        self.assertTrue(Month(200701).decrement(14).equal(Month(200511)))
        self.assertTrue(Month(200712).decrement().equal(Month(200711)))
        self.assertTrue(Month(200701).decrement(1).equal(Month(200612)))
        self.assertTrue(Month(200701).decrement(2).equal(Month(200611)))
        self.assertTrue(Month(200701).decrement(12).equal(Month(200601)))
        self.assertTrue(Month(200701).decrement(13).equal(Month(200512)))
        self.assertTrue(Month(200701).decrement(120).equal(Month(199701)))

if __name__ == '__main__':
    unittest.main()
    if False:
        pdb.set_trace()
