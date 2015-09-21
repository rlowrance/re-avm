# functions to parse the command line

import pdb
import unittest


class ParseCommandLine(object):
    def __init__(self, argv):
        self.argv = argv

    def default(self, tag, default_value):
        actual = self.get_arg(tag)
        return default_value if actual is None else actual

    def get_arg(self, tag):
        '''return value or list of values past the tag, or None

        --tag v1 --tag        ==> return v1 as a string
        --tag v1 v2 v3 --tag  ==> return v1 v2 v3 as list of string
        --tag v1 v2 v3        ==> return v1 v2 v3 as list of string
        --tag --tag           ==> return []
                              ==> return None
        '''
        for i in xrange(len(self.argv)):
            if self.argv[i] == tag:
                result = []
                i += 1
                while i < len(self.argv) and self.argv[i][:2] != '--':
                    result.append(self.argv[i])
                    i += 1
                return result[0] if len(result) == 1 else result
        return None

    def get_range(self, tag):
        '''return Python style range as list of ints, or list of strings
        --tag int1 ==> return [int1]
        --tag start stop ==> return [start, start+1, ..., stop-1]
        --tag start stop step ==> return [start, start+step, ...]
        --tag s1 ... sN == return [s1, ..., sN]
        '''
        def maybe_to_int(values):
            try:
                return [int(value) for value in values]
            except:
                msg = 'in range, one value in %s is not an int' % values
                print msg
                assert False, msg

        value = self.get_arg(tag)
        if value is None:
            return None
        if isinstance(value, str):
            return [int(value)]
        value_int = maybe_to_int(value)
        if isinstance(value, list):
            if len(value) == 1:
                return [value_int[0]]
        if len(value) == 2:
            return range(value_int[0], value_int[1])
        if len(value) == 3:
            return range(value_int[0], value_int[1], value_int[2])
        print 'cannot parse %s from %s' % (tag, self.argv)

    def has_arg(self, tag):
        'return True iff argv contains the tag'
        for i in xrange(len(self.argv)):
            if self.argv[i] == tag:
                return True
        return False


class TestParseCommandLine(unittest.TestCase):
    def setUp(self):
        argv = '--a --b bvalue --c 1 10 2 --d d1 d2'.split(' ')
        self.pcl = ParseCommandLine(argv)

    def test_default(self):
        self.assertEqual(self.pcl.default('--b', 'bdefault'), 'bvalue')
        self.assertEqual(self.pcl.default('--e', 'edefault'), 'edefault')

    def test_get_arg(self):
        self.assertEqual(self.pcl.get_arg('--b'), 'bvalue')
        self.assertEqual(self.pcl.get_arg('--c'), ['1', '10', '2'])
        self.assertEqual(self.pcl.get_arg('--d'), ['d1', 'd2'])

    def test_get_range(self):
        self.assertEqual(self.pcl.get_range('--c'), [1, 3, 5, 7, 9])

    def test_has_arg(self):
        self.assertTrue(self.pcl.has_arg('--a'))
        self.assertFalse(self.pcl.has_arg('--e'))


if __name__ == '__main__':
    if False:
        pdb.set_trace()
    unittest.main()
