from ColumnsTable import ColumnsTable
from Report import Report


class ReportWithColumnsTable(object):
    def __init__(self, header_lines, column_defs, print_as_spaces, verbose=True):
        self._report = Report()
        self._header(header_lines)
        self._ct = ColumnsTable(column_defs, verbose)
        self._print_as_spaces = print_as_spaces

    def _header(self, header_lines):
        for line in header_lines:
            self._report.append(line)

    def append_detail(self, **kwds):
        # replace NaN with None
        with_spaces = {k: (None if self._print_as_spaces(k, v) else v)
                       for k, v in kwds.iteritems()
                       }
        self._ct.append_detail(**with_spaces)

    def write(self, path):
        self._t.append_legend()
        for line in self._t.iterlines():
            self._report.append(line)
        self._report.write(path)
