'''create charts showing median and mean prices each month

INVOCATION
  python chart01.py [--data] [--test]

INPUT FILES
 INPUT/interesting_cities.txt
 INPUT/samples-train.csv

OUTPUT FILES
 WORKING/chart01/0data.pickle   # pd.DataFrame with columns date, month, city, price
 WORKING/chart01/median-price.pdf
 WORKING/chart01/median-price.txt
 WORKING/chart01/median-price_2006_2007.txt
 WORKING/chart01/prices-global.pdf
 WORKING/chart01/prices-cities.pdf
 WORKING/chart01/prices-city-{cityname}.pdf
'''

from __future__ import division

import argparse
import cPickle as pickle
import datetime
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pdb
from pprint import pprint
import random
import sys

from Bunch import Bunch
from ColumnsTable import ColumnsTable
from columns_contain import columns_contain
from Logger import Logger
from Month import Month
from Path import Path
from Report import Report
import layout_transactions as t
cc = columns_contain


def make_control(argv):
    # return a Bunch

    print argv
    parser = argparse.ArgumentParser()
    parser.add_argument('invocation')
    parser.add_argument('--data', help='reduce input and create data file in WORKING', action='store_true')
    parser.add_argument('--test', help='set internal test flag', action='store_true')
    arg = Bunch.from_namespace(parser.parse_args(argv))
    base_name = arg.invocation.split('.')[0]

    random_seed = 123
    random.seed(random_seed)

    dir_working = Path().dir_working()

    debug = False

    reduced_file_name = '0data.pickle'

    # assure output directory exists
    def assure_exists(dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        return dir_path

    dir_path = assure_exists(dir_working + 'chart01/')
    dir_date_price = assure_exists(dir_path + 'date-price/')
    dir_median_price = assure_exists(dir_path + 'median-price/')

    return Bunch(
        arg=arg,
        base_name=base_name,
        debug=debug,
        path_in_interesting_cities=dir_working + 'interesting_cities.txt',
        path_in_samples=dir_working + 'samples-train.csv',
        path_out_median_price=dir_median_price,
        path_out_date_price=dir_date_price,
        path_out_price_statistics_city_name=dir_path + 'price-statistics-city-name.txt',
        path_out_price_statistics_count=dir_path + 'price-statistics-count.txt',
        path_out_price_statistics_median_price=dir_path + 'price-statistics-median-price.txt',
        path_out_price_volume=dir_path + 'price-volume.pdf',
        path_out_stats_all=dir_path + 'price-stats-all.txt',
        path_out_stats_2006_2008=dir_path + 'price-stats-2006-2008.txt',
        path_reduction=dir_path + reduced_file_name,
        random_seed=random_seed,
        test=arg.test,
    )


def make_chart_price_volume(data, control):
    'Write pdf'
    def make_prices_volumes(data):
        'return tuples of dict[(year,month)] = number'
        def make_months(year):
            if year == 2009:
                return (1, 2, 3)
            else:
                return (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)

        prices = {}
        volumes = {}
        for year in (2003, 2004, 2005, 2006, 2007, 2008, 2009):
            for month in make_months(year):
                data_for_month = data[make_reduction_key(year, month)]
                price = data_for_month['median']
                volume = data_for_month['count']
                prices[(year, month)] = price
                volumes[(year, month)] = volume
        return prices, volumes

    prices, volumes = make_prices_volumes(data)  # 2 dicts

    def make_months(year):
            if year == 2009:
                return (1, 2, 3)
            else:
                return (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)

    years = (2003, 2004, 2005, 2006, 2007, 2008, 2009)
    months = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    months_2009 = (1, 2, 3)
    year_month = ['%s-%02d' % (year, month)
                  for year in years
                  for month in (months_2009 if year == 2009 else months)]
    figx = range(len(year_month))
    fig1y = []
    for year in (2003, 2004, 2005, 2006, 2007, 2008, 2009):
        for month in make_months(year):
            fig1y.append(prices[(year, month)])
    fig2y = []
    for year in (2003, 2004, 2005, 2006, 2007, 2008, 2009):
        for month in make_months(year):
            fig2y.append(volumes[(year, month)])

    fig = plt.figure()
    fig1 = fig.add_subplot(211)
    fig1.plot(figx, fig1y)
    x_ticks = [year_month[i] if i % 12 == 0 else ' '
               for i in xrange(len(year_month))
               ]
    plt.xticks(range(len(year_month)),
               x_ticks,
               # pad=8,
               size='xx-small',
               rotation=-30,
               # rotation='vertical',
               )
    plt.yticks(size='xx-small')
    plt.xlabel('year-month')
    plt.ylabel('median price ($)')
    plt.ylim([0, 700000])
    fig2 = fig.add_subplot(212)
    fig2.bar(figx, fig2y)
    plt.xticks(range(len(year_month)),
               x_ticks,
               # pad=8,
               size='xx-small',
               rotation=-30,
               # rotation='vertical',
               )
    plt.yticks(size='xx-small')
    plt.xlabel('year-month')
    plt.ylabel('number of sales')
    plt.savefig(control.path_out_price_volume)
    plt.close()


def make_reduction_key():
    pass  # stub


def make_charts_median_price(data, control):
    'write pdf file'
    # fix error in processing all cities at once: exceeded cell block limit in savefig() call below
    mpl.rcParams['agg.path.chunksize'] = 2000000  # fix error: exceeded cell block limit in savefig() call below

    def make_year_months():
        result = []
        for year in (2003, 2004, 2005, 2006, 2007, 2008, 2009):
            months = (1, 2, 3) if year == 2009 else (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
            for month in months:
                result.append((year, month))
        return result

    def fill_ax(ax, months, prices, title):
        'mutate axes ax'
        def median_price_for_year_month(year, month):
            in_year_month = months == Month(year, month)
            #  assert sum(in_year_month) > 0, (year, month)
            return prices[in_year_month].median()  # return NaN if no transactions in the (year,month)

        def x_tick_value(year, month):
            return '%4d-%02d' % (year, month) if month == 1 else ' '

        year_months = make_year_months()
        x = range(len(year_months))
        y = [median_price_for_year_month(year, month) for year, month in year_months]
        ax.plot(x, y)
        x_ticks = [x_tick_value(year, month) for year, month in year_months]
        plt.xticks(x, x_ticks, size='xx-small', rotation=-30)
        plt.yticks(size='xx-small')
        ax.set_xlabel('year-month')
        ax.set_ylabel('median price')
        if False:
            # impose the same y axis scale on each city
            # NOTE: An earlier version used the same y scale for each city
            # However, the largest median price in one month is about $10,000,000
            # and that was much larger than typical, so each city chart has its
            # own y scale.
            ylim_value = 10e6  # an upper bound on the largest median value
            ys = np.array(y)
            largest_y = ys[~ np.isnan(ys)].max()
            if largest_y > ylim_value:
                print largest_y, ylim_value
                pdb.set_trace()
            assert largest_y <= ylim_value, (y, largest_y, ylim_value)
            ax.set_ylim([0, ylim_value])
        ax.set_ylim(ymin=0)
        ax.set_title(title)

    def make_figure(months, prices, title):
        fig, ax = plt.subplots(1, 1)
        fill_ax(ax, months, prices, title)
        return fig

    def write_fig(fig, city):
        fig.savefig(control.path_out_median_price + city + '.pdf', format='pdf')

    for city in set(data.city):
        in_city = data.city == city
        local = data[in_city]
        print city, len(local)
        fig = make_figure(local.month, local.price, city)
        write_fig(fig, city)
        plt.close(fig)

    # process all cities at once
    fig = make_figure(data.month, data.price, 'all cities')
    write_fig(fig, 'all-cities')
    plt.close(fig)


def make_chart_stats(data, control, filter_f):
    'return Report with statistics for years and months that obey the filter'
    r = Report()
    r.append('Prices by Month')
    r.append('')
    ct = ColumnsTable((
            ('year', 4, '%4d', (' ', ' ', 'year'), 'year of transaction'),
            ('month', 5, '%5d', (' ', ' ', 'month'), 'month of transaction'),
            ('mean_price', 6, '%6.0f', (' ', ' mean', 'price'), 'mean price in dollars'),
            ('median_price', 6, '%6.0f', (' ', 'median', 'price'), 'median price in dollars'),
            ('mean_price_ratio', 6, '%6.3f', (' mean', ' price', ' ratio'), 'ratio of price in current month to prior month'),
            ('median_price_ratio', 6, '%6.3f', ('median', ' price', ' ratio'), 'ratio of price in current month to prior month'),
            ('number_trades', 6, '%6d', ('number', 'of', 'trades'), 'number of trades in the month'),
            ))

    prior_mean_price = None
    prior_median_price = None
    for year in xrange(2003, 2010):
        for month in xrange(1, 13):
            if filter_f(year, month):
                value = data[make_reduction_key(year, month)]
                mean_price = value['mean']
                median_price = value['median']
                number_trades = value['count']
                ct.append_detail(
                        year=year,
                        month=month,
                        mean_price=mean_price,
                        median_price=median_price,
                        mean_price_ratio=None if prior_mean_price is None else mean_price / prior_mean_price,
                        median_price_ratio=None if prior_median_price is None else median_price / prior_median_price,
                        number_trades=number_trades,
                        )
                prior_mean_price = mean_price
                prior_median_price = median_price
    ct.append_legend()
    for line in ct.iterlines():
        r.append(line)
    return r


def make_chart_stats_all(data, control):
    def filter_f(year, month):
        if year == 2009:
            return 1 <= month <= 3
        else:
            return True

    r = make_chart_stats(data, control, filter_f)
    r.write(control.path_out_stats_all)


def make_chart_stats_2006_2008(data, control):
    def filter_f(year, month):
        return year in (2006, 2007, 2008)

    r = make_chart_stats(data, control, filter_f)
    r.write(control.path_out_stats_2006_2008)


def make_charts_prices(data, control):
    'write output files'
    pdb .set_trace()
    pass


def make_charts_price_statistics(data, control):
    def make_column_table(cities, data):
        def append_detail_line(ct, city_name, prices):
            ct.append_detail(
                city=city_name,
                mean=prices.mean(),
                median=prices.median(),
                stddev=prices.std(),
                count=len(prices),
                )

        ct = ColumnsTable((
            ('city', 30, '%30s', ('city'), 'name of city'),
            ('mean', 7, '%7.0f', ('mean'), 'mean price across time periods'),
            ('median', 7, '%7.0f', ('median'), 'median price across time periods'),
            ('stddev', 7, '%7.0f', ('stddev'), 'standard deviation of prices across time periods'),
            ('count', 7, '%7.0f', ('count'), 'number of transactions across time periods'),
            ))
        for city in cities:
            in_city = data.city == city
            city_data = data[in_city]
            prices = city_data.price
            append_detail_line(ct, city, prices)

        # summary line is across all the cities
        append_detail_line(ct, '* all cities *', data.price)

        ct.append_legend()

        return ct

    def make_report(data, cities, sorted_by_tag):
        'return a Report'
        r = Report()
        r.append('Price Statistics by City')
        r.append('Sorted by %s' % sorted_by_tag)
        r.append('Transactions from %s to %s' % (data.date.min(), data.date.max()))
        r.append(' ')
        ct = make_column_table(cities, data)
        for line in ct.iterlines():
            r.append(line)
        return r

    def write_file(data, path_out, cities, sorted_by_tag):
        r = make_report(data, cities, sorted_by_tag)
        r.write(path_out)

    city_names = set(data.city)

    def in_city(city_name):
        'return DataFrame'
        is_in_city = data.city == city_name
        return data[is_in_city]

    def cities_by_city_name():
        'return iterable of cities orderd by city name'
        result = sorted(city_names)
        return result

    def cities_by_count():
        'return iterable of cities orderd by count'
        result = sorted(city_names, key=lambda city_name: len(in_city(city_name)))
        return result

    def cities_by_median_price():
        'return iterable of cities orderd by median_price'
        result = sorted(city_names, key=lambda city_name: in_city(city_name).price.median())
        return result

    write_file(data, control.path_out_price_statistics_city_name, cities_by_city_name(), 'City Name')
    write_file(data, control.path_out_price_statistics_count, cities_by_count(), 'Count')
    write_file(data, control.path_out_price_statistics_median_price, cities_by_median_price(), 'Median Price')


def make_data(control):
    'return DataFrame'
    def to_datetime_date(x):
        year = int(x / 10000.0)
        x -= year * 10000.0
        month = int(x / 100.0)
        x -= month * 100
        day = int(x)
        return datetime.date(year, month, day)

    transactions = pd.read_csv(control.path_in_samples,
                               nrows=10 if control.test else None,
                               )

    dates = [to_datetime_date(x) for x in transactions[t.sale_date]]
    months = [Month(date.year, date.month) for date in dates]

    result = pd.DataFrame({
        'price': transactions[t.price],
        'city': transactions[t.city],
        'date': dates,
        'month': months,
        })
    return result


def read_interesting_cities(path_in):
    'return lines in input file'
    pdb.set_trace()
    result = []
    with open(path_in) as f:
        for line in f:
            result.append(line)
    return result


def make_charts_date_price(data, control):
    'write files with x-y plots for date and price'

    # fix error in processing all cities at once: exceeded cell block limit in savefig() call below
    mpl.rcParams['agg.path.chunksize'] = 2000000  # fix error: exceeded cell block limit in savefig() call below

    def fill_ax(ax, x, y, title, xlabel, ylabel, ylimit):
        'mutate axes ax'
        ax.plot(x, y, 'bo')
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title('%s N=%d' % (title, len(x)))
        ymax = y.max() + 1 if ylimit is None else ylimit
        ax.set_ylim([0, ymax])

    def make_figure(x, y, title):
        print 'make_figure', title
        fig, ax = plt.subplots(2, 1)

        uniform_scaling = False
        logymax = int(math.log10(y.max()) + 1) if uniform_scaling else None
        ymax = 10 ** logymax if uniform_scaling else None

        fill_ax(ax[0], x, y, title, None, 'price', ymax)
        fill_ax(ax[1], x, np.log10(y), None, 'sale date', 'log10(price)', logymax)

        return fig

    for city in set(data.city):
        in_city = data.city == city
        local = data[in_city]
        fig = make_figure(local.date, local.price, city)
        fig.savefig(control.path_out_date_price + city + '.pdf', format='pdf')
        plt.close(fig)

    # process all cities at once
    fig = make_figure(data.date, data.price, 'all cities')
    fig.savefig(control.path_out_date_price + 'all-cities', format='pdf')
    plt.close(fig)


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(base_name=control.base_name)
    print control

    if control.arg.data:
        data = make_data(control)
        with open(control.path_reduction, 'wb') as f:
            pickle.dump((data, control), f)
    else:
        # interesting_cities = read_interesting_cities(control.path_in_interesting_cities)
        with open(control.path_reduction, 'rb') as f:
            data, reduction_control = pickle.load(f)
        make_charts_median_price(data, control)
        print 'TESTING'
        return
        make_charts_date_price(data, control)
        make_charts_price_statistics(data, control)
        make_chart_price_volume(data, control)
        make_chart_stats_2006_2008(data, control)
        make_chart_stats_all(data, control)

    print control
    if control.test:
        print 'DISCARD OUTPUT: test'
    print 'done'

    return


if __name__ == '__main__':
    if False:
        # avoid pyflakes warnings
        pdb.set_trace()
        pprint()
        pd.DataFrame()
        np.array()

    main(sys.argv)
