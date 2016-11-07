
from __future__ import division

import matplotlib.pyplot as plt
import pdb

from ColumnsTable import ColumnsTable
from columns_contain import columns_contain
from Month import Month
from Report import Report
cc = columns_contain


class ChartCDReport(object):
    def __init__(self, column_definitions, test):
        self._column_definitions = column_definitions
        self._test = test
        self._report = Report()
        cd = self._column_definitions.defs_for_columns(
            'validation_month', 'rank', 'median_absolute_error',
            'median_price', 'model', 'n_months_back',
            'max_depth', 'n_estimators', 'max_features',
            'learning_rate', 'alpha', 'l1_ratio',
            'units_X', 'units_y',
        )
        self._ct = ColumnsTable(columns=cd, verbose=True)
        self._header()

    def append(self, line):
        self._report.append(line)

    def write(self, path):
        self._ct.append_legend()
        for line in self._ct.iterlines():
            self._report.append(line)
        if self._test:
            self._report.append('** TESTING: DISCARD')
        self._report.write(path)

    def _header(self):
        self._report.append('Median Absolute Error (MAE) by month for best-performing models and their hyperparameters')
        self._report.append(' ')

    def append_detail(self, **kwds):
        with_spaces = {
            k: (None if self._column_definitions.replace_by_spaces(k, v) else v)
            for k, v in kwds.iteritems()
        }
        self._ct.append_detail(**with_spaces)


def make_chart_cd(reduction, median_prices, control, detail_line_indices, report_id):
    r = ChartCDReport(control.column_definitions, control.test)
    my_validation_months = []
    my_price = []
    my_mae = []
    for validation_month in control.validation_months_long:
        median_price = median_prices[Month(validation_month)]

        if validation_month not in reduction:
            control.exceptions.append('reduction is missing month %s' % validation_month)
            continue
        month_result_keys = reduction[validation_month].keys()
        my_validation_months.append(validation_month)
        my_price.append(median_price)
        for detail_line_index in detail_line_indices:
            if detail_line_index >= len(month_result_keys):
                continue  # this can happend when using samples
            try:
                k = month_result_keys[detail_line_index]
            except:
                pdb.set_trace()
            k = month_result_keys[detail_line_index]
            v = reduction[validation_month][k]
            r.append_detail(
                validation_month=validation_month,
                rank=detail_line_index + 1,
                median_absolute_error=v.mae,
                median_price=median_price,
                model=k.model,
                n_months_back=k.n_months_back,
                max_depth=k.max_depth,
                n_estimators=k.n_estimators,
                max_features=k.max_features,
                learning_rate=k.learning_rate,
                alpha=k.alpha,
                l1_ratio=k.l1_ratio,
                units_X=k.units_X[:3],
                units_y=k.units_y[:3],
            )
        my_mae.append(reduction[validation_month][month_result_keys[0]].mae)

    fig = plt.figure()
    fig1 = fig.add_subplot(211)
    fig1.bar(range(len(my_validation_months)), my_mae, color='blue')
    labels = my_validation_months
    plt.xticks([x+.6 for x in range(len(my_validation_months))], labels, rotation=-70, size='xx-small')

    plt.yticks(size='xx-small')
    plt.xlabel('Year-Month')
    plt.ylabel('Median Absolute Error ($)')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    fig2 = fig.add_subplot(212)
    fig2.bar(
        range(len(my_validation_months)),
        [int(m) / int(p) for m, p in zip(my_mae, my_price)],
        color='blue',
        )
    plt.xticks([
        x+.6
        for x in range(len(my_validation_months))
        ],
        labels,
        rotation=-70,
        size='xx-small',
        )

    plt.yticks(size='xx-small')
    plt.xlabel('Year-Month')
    plt.ylabel('Absolute Relative Error')

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    plt.savefig(control.path_out_c_pdf)
    plt.close()

    r.write(control.path_out_cd % report_id)
    return
