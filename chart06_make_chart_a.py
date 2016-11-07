
from __future__ import division

import matplotlib.pyplot as plt
import pdb

from columns_contain import columns_contain
from Month import Month
cc = columns_contain


def make_chart_a(reduction, median_prices, control):
    'graph range of errors by month by method'
    print 'make_chart_a'

    def make_subplot(validation_month, reduction, relevant_median_prices):
        'mutate the default axes'
        # draw one line for each model family
        for model in ('en', 'gb', 'rf'):
            y = [v.mae
                 for k, v in reduction[validation_month].iteritems()
                 if k.model == model
                 ]
            plt.plot(y, label=model)  # the reduction is sorted by increasing mae
            plt.yticks(size='xx-small')
            if Month(validation_month) not in relevant_median_prices:
                print validation_month
                print relevant_median_prices
                print 'should not happen'
                pdb.set_trace()
            plt.title('yr mnth %s med price %6.0f' % (
                validation_month,
                relevant_median_prices[Month(validation_month)]),
                      loc='right',
                      fontdict={'fontsize': 'xx-small',
                                'style': 'italic',
                                },
                      )
            plt.xticks([])  # no ticks on x axis
        return

    def make_figure(reduction, path_out, city, relevant_median_prices):
        # make and save figure

        # debug: sometimes relevant_median_prices is empty
        if len(relevant_median_prices) == 0:
            print 'no median prices', city
            pdb.set_trace()

        plt.figure()  # new figure
        # plt.suptitle('Loss by Test Period, Tree Max Depth, N Trees')  # overlays the subplots
        axes_number = 0
        validation_months = ('200612', '200701', '200702', '200703', '200704', '200705',
                             '200706', '200707', '200708', '200709', '200710', '200711',
                             )
        row_seq = (1, 2, 3, 4)
        col_seq = (1, 2, 3)
        cities = city is not None
        for row in row_seq:
            for col in col_seq:
                validation_month = validation_months[axes_number]
                if cities:
                    print 'city %s validation_month %s num transactions %d' % (
                        city,
                        validation_month,
                        len(reduction[validation_month]))
                axes_number += 1  # count across rows
                plt.subplot(len(row_seq), len(col_seq), axes_number)  # could be empty, if no transactions in month
                make_subplot(validation_month, reduction, relevant_median_prices)
                # annotate the bottom row only
                if row == 4:
                    if col == 1:
                        plt.xlabel('hp set')
                        plt.ylabel('mae x $1000')
                    if col == 3:
                        plt.legend(loc='best', fontsize=5)

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(path_out)
        plt.close()

    if control.arg.locality == 'global':
        make_figure(reduction, control.path_out_a, None, median_prices)
    elif control.arg.locality == 'city':

        def make_city(city):
            print 'make_city', city
            assert len(reduction[city]) > 0, city  # detect bug found in earlier version
            return make_figure(reduction[city], control.path_out_a % city, city, median_prices[city])

        for city in reduction.keys():
            make_city(city)
    else:
        print 'bad control.arg.locality', control.arg
        pdb.set_trace()
    return
