
from __future__ import division

import itertools
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from ColumnsTable import ColumnsTable
from Month import Month
from Report import Report


class ChartBReport(object):
    def __init__(self, validation_month, k, column_definitions, test):
        self._report = Report()
        self._header(validation_month, k)
        self._column_definitions = column_definitions
        self._test = test
        cd = self._column_definitions.defs_for_columns(
            'median_absolute_error', 'model', 'n_months_back',
            'max_depth', 'n_estimators', 'max_features',
            'learning_rate',
        )
        self._ct = ColumnsTable(columns=cd, verbose=True)

    def _header(self, validation_month, k):
        def a(line):
            self._report.append(line)

        a('MAE for %d best-performing models and their hyperparameters' % k)
        a('Validation month: %s' % validation_month)
        a(' ')

    def append_detail(self, **kwds):
        # replace NaN with None
        with_spaces = {
            k: (None if self._column_definitions.replace_by_spaces(k, v) else v)
            for k, v in kwds.iteritems()
        }
        self._ct.append_detail(**with_spaces)

    def write(self, path):
        self._ct.append_legend()
        for line in self._ct.iterlines():
            self._report.append(line)
        if self._test:
            self._report.append('**TESTING: DISCARD')
        self._report.write(path)


def make_chart_b(reduction, control, median_price):
    def make_models_maes(validation_month):
        'return model names and MAEs for K best models in the valdation month'
        k = 50  # report on the first k models in the sorted subset
        # ref: http://stackoverflow.com/questions/7971618/python-return-first-n-keyvalue-pairs-from-dict
        first_k_items = itertools.islice(reduction[validation_month].items(), 0, k)
        graphX = []
        graphY = []
        for key, value in first_k_items:
            graphY.append(value.mae)
            graphX.append(key.model)

        return graphX, graphY

    def make_figure():
        'make and write figure'
        plt.figure()  # new figure
        validation_months = control.validation_months
        row_seq = (1, 2, 3, 4)
        col_seq = (1, 2, 3)
        axes_number = 0
        for row in row_seq:
            for col in col_seq:
                validation_month = validation_months[axes_number]
                axes_number += 1  # count across rows
                ax1 = plt.subplot(len(row_seq), len(col_seq), axes_number)
                graphX, graphY = make_models_maes(validation_month)
                patterns = ["", "", "*"]
                # the reduction is sorted by increasing mae
                # Jonathan
                ax1.set_title(
                    'Validation Month: %s' % (validation_month),
                    loc='right',
                    fontdict={'fontsize': 'xx-small', 'style': 'italic'},
                    )
                for i in range(len(graphX)):
                    if graphX[i] == 'gb':
                        plt.bar(i, graphY[i], color='white', edgecolor='black', hatch=patterns[0])
                    elif graphX[i] == 'rf':
                        plt.bar(i, graphY[i], color='black', edgecolor='black', hatch=patterns[1])
                    elif graphX[i] == 'en':
                        plt.bar(i, graphY[i], color='green', edgecolor='black', hatch=patterns[2])
                plt.yticks(size='xx-small')
                plt.xticks([])

                # annotate the bottom row only
                if row == 4:
                    if col == 1:
                        plt.xlabel('Models')
                        plt.ylabel('MAE')
                    if col == 3:
                        plt.legend(loc='best', fontsize=5)

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(control.path_out_b_pdf_subplots)
        plt.close()

    def make_figure2(validation_month):
        '''make and write figure for the validation month
        Part 1:
        for the validation month
        one bar for each of the first 50 best models
        the height of the bar is the MAE in ($)
        Part 2:
        produce a 2-up chart, where the top chart is as in part 1
        and the bottom chart has as y axis the absolute relative error
        '''

        print 'creating figure b', validation_month

        # plt.suptitle('Loss by Test Period, Tree Max Depth, N Trees')  # overlays the subplots
        bar_color = {'gb': 'white', 'rf': 'black', 'en': 'red'}
        models, maes = make_models_maes(validation_month)
        assert len(models) == len(maes)
        assert len(models) > 0
        # the reduction is sorted by increasing mae
        # Jonathan
        fig = plt.figure()
        fig1 = fig.add_subplot(211)

        plt.title(
            'Validation Month: %s' % (validation_month),
            loc='right',
            fontdict={'fontsize': 'large', 'style': 'italic'},
            )
        for i, model in enumerate(models):
            fig1.bar(i, maes[i], color=bar_color[model])
        plt.yticks(size='xx-small')
        plt.xticks([])
        plt.xlabel('Models in order of increasing MAE')
        plt.ylabel('MAE ($)')

        white_patch = mpatches.Patch(
            facecolor='white',
            edgecolor='black',
            lw=1,
            label="Gradient Boosting",
            )
        black_patch = mpatches.Patch(
            facecolor='black',
            edgecolor='black',
            lw=1,
            label="Random Forest",
            )

        plt.legend(handles=[white_patch, black_patch], loc=2)
        plt.ylim(0, 180000)

        fig2 = fig.add_subplot(212)
        for i, model in enumerate(models):
            fig2.bar(i, maes[i]/median_price[Month(validation_month)], color=bar_color[model])

        plt.yticks(size='xx-small')
        plt.xticks([])
        plt.xlabel('Models in order of increasing MAE')
        plt.ylabel('Absolute Relative Error')
        plt.ylim(0, .3)

        white_patch = mpatches.Patch(
            facecolor='white',
            edgecolor='black',
            lw=1,
            label="Gradient Boosting",
            )
        black_patch = mpatches.Patch(
            facecolor='black',
            edgecolor='black',
            lw=1,
            label="Random Forest",
            )

        plt.legend(handles=[white_patch, black_patch], loc=2)
        plt.savefig(control.path_out_b_pdf % int(validation_month))
        plt.close()

    # produce the pdf files
    for validation_month in control.validation_months:  # TODO: validation_month_long
        make_figure2(validation_month)
    make_figure()

    def write_report(year, month):
        validation_month = str(year * 100 + month)
        k = 50  # report on the first k models in the sorted subset
        report = ChartBReport(validation_month, k, control.column_definitions, control.test)
        detail_line_number = 0
        # ref: http://stackoverflow.com/questions/7971618/python-return-first-n-keyvalue-pairs-from-dict
        first_k = itertools.islice(reduction[validation_month].items(), 0, k)
        graphX = []
        graphY = []
        for key, value in first_k:
            report.append_detail(
                median_absolute_error=value.mae,
                model=key.model,
                n_months_back=key.n_months_back,
                max_depth=key.max_depth,
                n_estimators=key.n_estimators,
                max_features=key.max_features,
                learning_rate=key.learning_rate,
            )
            graphX.append(value.mae)
            graphY.append(key.model)
            detail_line_number += 1
            if detail_line_number > k:
                break
        report.write(control.path_out_b % int(validation_month))

    # produce the txt file
    for year in (2006, 2007):
        months = (12,) if year == 2006 else (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
        for month in months:
            write_report(year, month)
