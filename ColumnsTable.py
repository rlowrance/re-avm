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
            # convert 'abc' to ('abc',)
            headers = (cd3,) if isinstance(cd3, str) else cd3
            if self._number_of_header_lines != 0:
                if self._number_of_header_lines != len(headers):
                    print 'inconsistent number of lines in header'
                    print 'have both %d and %d' % (self._number_of_header_lines, len(headers))
                    print 'found at headesr: %s' % headers
                    pdb.set_trace()
            self._number_of_header_lines = len(headers)
            self._column_defs.append(ColumnsTableFields(column_def[0],
                                                        column_def[1],
                                                        column_def[2],
                                                        headers,
                                                        column_def[4]))
        self._lines = []
        self._verbose = verbose
        self._header()

    def append_legend(self):
        def cat_headers(headers):
            text = ''
            for header in headers:
                if len(text) > 0:
                    text += ' '
                text += header.strip()
            return text

        self._append_line(' ')
        self._append_line('column legend:')
        for cd in self._column_defs:
            line = '%12s -> %s' % (cat_headers(cd.headers), cd.legend)
            self._append_line(line)

    def iterlines(self):
        for line in self._lines:
            yield line

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
        self._append_line(line)

    def _append_line(self, line):
        if self._verbose:
            print line
        self._lines.append(line)

    def _header(self):
        def append_header(index):
            line = ''
            for cd in self._column_defs:
                formatter = '%' + str(cd.width) + 's'
                formatted = formatter % cd.headers[index]
                line += (' ' if len(line) > 0 else '') + formatted
            self._append_line(line)

        for header_line_index in xrange(self._number_of_header_lines):
            append_header(header_line_index)

    def _print(self):
        for line in self._lines:
            print line


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

    def test_append_detail_line(self):
        c = self.columns
        c.append_detail(a=10, bcd=20)
        c.append_detail(bcd=30)
        c.append_detail(a=50)
        self.assertEqual(len(c._lines), 5)
        self.assertTrue(True)

    def test_iterlines(self):
        c = self.columns
        counter = collections.Counter()

        for line in c.iterlines():
            counter['line'] += 1
            if self.verbose:
                print line

        self.assertEqual(counter['line'], 2)  # only 2 column headers lines

    def test_append_legend_lines(self):
        c = self.columns
        c.append_legend()
        self.assertTrue(True)

    def test_one_header(self):
        c = ColumnsTable(
            (('a', 10, '%d', 'first', 'words'),
             ('b', 20, '%d', 'second', 'more words'),
             )
        )
        self.assertEqual(c._number_of_header_lines, 1)


if __name__ == '__main__':
    unittest.main()
    if False:
        pdb
