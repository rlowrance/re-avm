'''Determine which models are most accurate

INVOCATION
 python rank_models.py

INPUTS
 WORKING/chart06/data.pickle  Defines the best models

OUTPUTS
 WORKING/rank_models/MONTH.pickle, containing dict
   reduction[month]=OrderedDict(ModelDescription, ModelResults)
'''

from __future__ import division

import argparse
import cPickle as pickle
import os
import pandas as pd
import pdb
from pprint import pprint as pp
import random
import sys

from AVM import AVM
from Bunch import Bunch
from chart06 import ModelDescription, ModelResults, ColumnDefinitions
from Path import Path
# from Report import Report
from Timer import Timer
# from valavm import ResultKeyEn, ResultKeyGbr, ResultKeyRfr, ResultValue
# cc = columns_contain


def make_control(argv):
    'return a Bunch'
    parser = argparse.ArgumentParser()
    parser.add_argument('invocation')
    parser.add_argument('--data', action='store_true')
    parser.add_argument('--test', action='store_true')
    arg = parser.parse_args(argv)
    arg.base_name = 'rank_models'

    random_seed = 123
    random.seed(random_seed)

    dir_working = Path().dir_working()
    dir_out = dir_working + arg.base_name + '/'
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    # fit models for these months
    months = (
        '200512',
        '200601', '200602', '200603', '200604', '200605', '200606',
        '200607', '200608', '200609', '200610', '200611', '200612',
        '200701', '200702', '200703', '200704', '200705', '200706',
        '200707', '200708', '200709', '200710', '200711', '200712',
        '200801', '200802', '200803', '200804', '200805', '200806',
        '200807', '200808', '200809', '200810', '200811', '200812',
        '200901', '200902', '200903',
    )

    return Bunch(
        arg=arg,
        debug=True,
        months=months,
        path_in_chart06=dir_working + 'chart06/data.pickle',
        path_out_dir=dir_out,
        timer=Timer(),
    )


def main(argv):
    control = make_control(argv)
    print control

    # read the input pickle file
    with open(control.path_in_chart06, 'rb') as f:
        print 'reading ranked models'
        pickled = pickle.load(f)
        pdb.set_trace()
        reduction, all_actuals, median_price, chart_06_control = pickled

    # write the monthly output files
    for month in control.months:
        if month in reduction:
            print 'saving', month
            path = control.path_out_dir + month + '.pickle'
            with open(path, 'wb') as f:
                pickle.dump(reduction[month], f)
        else:
            print 'month not in reduction from chart06:', month

    # wrap up
    print control
    if control.arg.test:
        print 'DISCARD OUTPUT: test'
    if control.debug:
        print 'DISCARD OUTPUT: debug'
    print 'done'


if __name__ == '__main__':
    if False:
        # avoid pyflakes warnings
        pdb.set_trace()
        pp()
        pd.DataFrame()
        ModelDescription
        ModelResults
        ColumnDefinitions
        AVM()

    main(sys.argv)
