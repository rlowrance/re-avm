'''create ColumnsTable that can be printed in a txt file

    c = ColumnsTable(('colname', 6, '%6.2f', ('header1', 'header2'), 'legend'),)
    c.append(colname=10)  # also other column name=value pairs
    ...
    c.append_legend()
    r = Report()
    r.append('title')
    c.iterate_lines(lambda line: r.append(line))  # also takes kwds
    r.write(path)
'''

import collections
import pdb
import unittest

ColumnsTableFields = collections.namedtuple('ColumnsTableFields', 'name width formatter headers legend')


class ColumnsTable(object):
    def __init__(self, column_defs, verbose=False):
        'column_defs is an iterable with elements (name, width, formatter, (header-list), legend)'
        self._column_defs = []
        self._number_of_header_lines = 0
        for column_def in column_defs:
            cd3 = column_def[3]
            header = cd3 if isinstance(cd3, list) else (cd3,)
            if self._number_of_header_lines != 0:
                if self._number_of_header_lines != len(header):
                    print 'inconsistent number of lines in header'
                    print 'have both %d and %d' % (self._number_of_header_lines, len(header))
                    print 'found at header: %s' % header
                    pdb.set_trace()
            self._number_of_header_lines = len(header)
            self._column_defs.append(ColumnsTableFields(column_def[0],
                                                        column_def[1],
                                                        column_def[2],
                                                        header,
                                                        column_def[4]))
        self._lines = []
        self._verbose = verbose
        self._header()

    def _print(self):
        for line in self._lines:
            print line

    def append(self, line):
        if self._verbose:
            print line
        self._lines.append(line)

    def append_legend(self):
        def cat_headers(headers):
            text = ''
            for header in headers:
                if len(text) > 0:
                    text += ' '
                text += header.strip()
            return text

        self.append(' ')
        self.append('column legend:')
        for cd in self._column_defs:
            line = '%12s -> %s' % (cat_headers(cd.headers), cd.legend)
            self.append(line)

    def iterate_lines(self, func, **kwds):
        for line in self._lines:
            func(line, kwds)

    def _header(self):
        def append_header(index):
            line = ''
            for cd in self._column_defs:
                formatter = '%' + str(cd.width) + 's'
                formatted = formatter % cd.headers[index]
                line += (' ' if len(line) > 0 else '') + formatted
            self.append(line)

        for header_line_index in xrange(self._number_of_header_lines):
            append_header(header_line_index)

    def append_detail(self, **kwds):
        line = ''
        for cd in self._column_defs:
            if cd.name in kwds:
                glyph = cd.formatter % kwds[cd.name]
            else:
                glyph = ' ' * cd.width
            if len(line) > 0:
                line += ' '
            line += glyph
        self.append(line)


class TestColumnsTable(unittest.TestCase):
    def setUp(self,):
        self.verbose = False
        self.columns = ColumnsTable(
            (('a', 3, '%3d', ('one', 'num'), 'angstroms'),
             ('bcd', 10, '%10.2f', ('length', 'meters'), 'something really big'),
             ),
            verbose=self.verbose,
        )

    def test_construction(self):
        self.assertTrue(True)
        pass  # tested in setUp()

    def test_append(self):
        c = self.columns
        c.append_detail(a=10, bcd=20)
        c.append_detail(bcd=30)
        c.append_detail(a=50)
        self.assertEqual(len(c._lines), 5)
        self.assertTrue(True)

    def test_iterate_lines(self):
        c = self.columns
        counter = collections.Counter()

        def handle(line, kwds):
            if self.verbose:
                print line
            kwds['counter']['line'] += 1

        c.iterate_lines(handle, counter=counter)
        self.assertEqual(counter['line'], 2)  # only 2 column headers lines

    def test_legend(self):
        c = self.columns
        c.append_legend()
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
    if False:
        pdb
