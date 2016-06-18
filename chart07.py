'''Determine most important features for the very best K models in each test month

valavm.py didn't save the fitted models, because that would have created a lot
of data.  So this program re-fits the model, in order to gain access to the
scikit-learn feature_importances_ attribute.

INVOCATION
 python chart07.py --data
  create WORKING/chart06/data.pickle
 python chart07.py
  create the actual charts TODO: define these

INPUTS
 WORKING/samples-train.csv    Training data needed to fit the models
 WORKING/chart07/data.pickle  Defines the best models

OUTPUTS
 WORKING/chart07/data.pickle
 WORKING/chart06/a.txt        TODO: define this
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
from Features import Features
from Path import Path
# from Report import Report
from Timer import Timer
# cc = columns_contain


def make_control(argv):
    'return a Bunch'
    parser = argparse.ArgumentParser()
    parser.add_argument('invocation')
    parser.add_argument('--data', action='store_true')
    parser.add_argument('--test', action='store_true')
    arg = parser.parse_args(argv)
    arg.base_name = 'chart07'

    random_seed = 123
    random.seed(random_seed)

    dir_working = Path().dir_working()
    dir_out = dir_working + arg.base_name + '/'
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    # fit models for these months
    test_months = (
        '200512',
        '200601', '200602', '200603', '200604', '200605', '200606',
        '200607', '200608', '200609', '200610', '200611', '200612',
        '200701', '200702', '200703', '200704', '200705', '200706',
        '200707', '200708', '200709', '200710', '200711', '200712',
        '200801', '200802', '200803', '200804', '200805', '200806',
        '200807', '200808', '200809', '200810', '200811', '200812',
        '200901', '200902',
    )
    reduced_file_name = 'data.pickle'

    return Bunch(
        arg=arg,
        debug=True,
        k=1,  # number of best models examined
        path_in_data=dir_out + reduced_file_name,
        path_in_fitted_dir=dir_working + 'valavm/',
        path_out_data=dir_out + reduced_file_name,
        path_out_a=dir_out + 'a.txt',
        test_months=test_months,
        timer=Timer(),
    )


def make_charts(control, importances):
    'return dict of charts'
    # all models are fit to an X matrix with the same features in the same columns
    pdb.set_trace()
    assert control.k == 1, control  # the codes works only for the very best model


def make_data(control):
    'return dict[k] = (model_type, coefficients_or_feature_importances)'
    feature_names = Features().ege_names()
    result = {}
    for test_month in control.test_months:
        path = control.path_in_fitted_dir + 'fitted-' + test_month + '.pickle'
        print 'make_data reading', path
        with open(path, 'rb') as f:
            # reduce only first k records in input
            for k in xrange(control.k):
                pickled = pickle.load(f)  # pickled is a tuple
                index, key, importances = pickled
                if index != k:
                    print index, k
                # we don't handle the coefficients from the en models
                # but no en model is among the best performing
                assert key.model == 'gb' or key.model == 'rf', key
                for index in xrange(len(importances)):
                    feature_name = feature_names[index]
                    importance = importances[index]
                    key = (feature_name, test_month)
                    assert key not in result, key
                    result[key] = importance
    return result


def main(argv):
    control = make_control(argv)
    print control

    # do the work
    if control.arg.data:
        data = make_data(control)
        control.timer.lap('make data reduction')
        with open(control.path_out_data, 'wb') as f:
            pickle.dump((data, control), f)
            control.timer.lap('write reduction')
    else:
        with open(control.path_in_data, 'rb') as f:
            pickled = pickle.load(f)
            data, reduction_control = pickled
        charts = make_charts(control, data)
        control.timer.lap('make charts')
        print charts
        # TODO: write the charts, which are a dictionary
        control.timer.lap('write charts')

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
