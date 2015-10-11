'''provide information to assess the importance of features in WORKING/samples-train.csv

Assess importance to sale price

Assess importance through
* mutual information
* Pearson's rho correlation coefficient

OUTPUT FILES
 WORKING/filter-features.txt
 WORKING/filter-features.pickle
'''

import collections
import cPickle as pickle
import pandas as pd
import pdb
import random
import sys

import Bunch
import layout_transactions as t
import Logger
import ParseCommandLine
import Path
from Report import Report


def usage(msg=None):
    if msg is not None:
        print msg
    print 'usage : python filter-features.py [--test]'
    print ' --test : run in test mode'
    sys.exit(1)


def make_control(argv):
    # return a Bunch
    print argv
    if len(argv) not in (2, 3):
        usage('invalid number of arguments')

    pcl = ParseCommandLine.ParseCommandLine(argv)
    arg = Bunch.Bunch(
        base_name=argv[0].split('.')[0],
        test=pcl.has_arg('--test'),
    )

    random_seed = 123456
    random.seed(random_seed)

    path = Path.Path()  # use the default dir_input

    debug = False

    out_file_base = (
        arg.base_name +
        ('-test' if arg.test else '')
    )

    return Bunch.Bunch(
        arg=arg,
        debug=debug,
        max_sale_price=85e6,  # according to Wall Street Journal
        path_in=path.dir_working() + 'samples-train.csv',
        path_out_txt=path.dir_working() + out_file_base + '.csv',
        path_out_pickle=path.dir_working() + out_file_base + '.pickle',
        random_seed=random_seed,
        test=arg.test,
    )


def make_mutual_info(df):
    'return dictionary key=feature value=I(t.price,feature)'
    # I(x,y) = \sum_x \sum_y p(x,y) log_2{ p(x,y)/p(x)p(y) }
    # The units are bit, since log_2 is used

    def prob1(xs):
        pdb.set_trace()
        p = collections.defaultdict(float)
        for x in xs:
            p[x] += 1.0 / len(xs)
        pdb.set_trace()
        return p

    def prob2(xs, ys):
        pdb.set_trace()
        assert len(xs) == len(ys), (len(xs), len(ys))
        p = collections.defaultdict(float)
        for i in xrange(len(xs)):
            x = xs[i]
            y = ys[i]
            p[(x, y)] += 1.0 / len(xs)
        pdb.set_trace()
        return p

    def prob(xs, ys=None):
        return prob1(xs) if ys == None else prob2(xs, ys)

    pdb.set_trace()
    prices_all = df[t.price]  # prices are in dollars, so are discrete
    p_prices_all = prob(prices_all)

    for feature_name in df.columns:
        series = df[feature_name]
        mask_present = series.notnull()
        series_present = series[mask_present]
        prices_present = prices_all[mask_present]
        p_x_y = prob(prices_present, series_present)
        p_x = prob(prices_present)
        p_y = prob(series_present)
        mutual_info = 0.0
        for x in prices_present:
            for y in series_present:
                fraction = p_x_y(x, y) / (p_x(x) * p_y(y))
                ln2 = 0.0 if fraction == 0 else math.log(fraction, 2)
                mututal_info += p_x_y(x, y) * ln2
        pdb.set_trace()


        # TODO: reject unless numeric or a code
        # TODO: remove samples where series values is missing
        p_x_y = probs(prices_all, series)


    for

    # determine p(x) = p(price)
    p_price = collections.defaultdict(float)
    for price in prices:
        p_price[price] += 1.0 / len(prices)

    d = {}
    for feature in df.columns:
        feature_value = df[feature]
        pass


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger.Logger(base_name=control.arg.base_name)
    print control

    in_df = pd.read_csv(control.path_in,
                        nrows=1000 if control.test else None,
                        )
    d_mutual_info = make_mutual_info(in_df)
    d_pearson = make_person(in_df)
    r = make_report(d_mutual_info, d_pearson)

    f = open(control.path_out_report, 'wb')
    pickle.dump((d_mutual_info, d_pearson, r, control), f)
    f.close()

    if control.test:
        print 'DISCARD OUTPUT: TESTING'

    print control
    print 'done'


if __name__ == '__main__':
    if False:
        pdb.set_trace()
        Report()
    main(sys.argv)
