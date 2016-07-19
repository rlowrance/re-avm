'''Determine accuracy on validation set YYYYMM of various hyperparameter setting
for AVMs based on 3 models (linear, random forests, gradient boosting regression)

INVOCATION
  python valavm.py {features_group}-{hps}-{locality}{-validation_month} \
                   [--test] [--renameoutput] [--makefile [{system} {threads} ...]]
  where
   features_group in {s, sw, swp, swpn}
     features to use
     s: just size (lot and house)
     sw: also weath (3 census track wealth features)
     swp: also property features (rooms, has_pool, ...)
     swpn: also neighborhood features (of census tract and zip5)
   hps in {all}
     hyperaparameters to sweep; possible values
     all: sweep all
   validation_month is of the form YYYYMM (year, month)
     month to use to create the estimated generalization error
     the training data are in the months just before this month.
   locality in {global, census, city, zip}
     if global, a single model is trained on all the data and used for every validation sample
     otherwise, a model is trained for each location and used to estimate every validation sample
       in that location
        if locality is census, a separate model is trained for each census tract
        if locality is city a separate model is trained for each city
        if locality is zip, a separate model is trained for each 5-digit zip code
   --test   : if present, runs in test mode, output is not usable
   --renameoutput
     change output file names to conform to new output file scheme
     from WORKING/valavm/{features_group}-{hps}/{features_group}-{hps}-{month}.pickle
     to   WORKING/valavm/{features_group}-{hps}-{city}/{month}.pickle
   --makefile
     create valavm.makefile containing rules that make valavm outputs on the specified
     {system}s each of which has the specified number of {threads}.
     Default arg is 'dell 16 roy 12 judith 7 hp 4'

INPUTS
 WORKING/samples-train.csv

OUTPUTS
 working/valavm/{features_group}-{hps}-{locality}/{validation_month}.pickle
 SRC/valavm.makefile

NOTE 1
The codes for the FEATURES are used directly in AVM and Features, so if you
change them here, you must also change then in those two modules.

NOTE 2
The output is in a directory instead of a file so that Dropbox's selective sync
can be used to control the space used on systems.
'''

from __future__ import division

import argparse
import collections
import cPickle as pickle
import itertools
import numpy as np
import os
import pandas as pd
import pdb
from pprint import pprint
import random
import sys

import arg_type
import AVM
from Bunch import Bunch
from columns_contain import columns_contain
# from Features import Features
import layout_transactions
from Logger import Logger
from Month import Month
from Path import Path
from Report import Report
from SampleSelector import SampleSelector
from Timer import Timer
# from TimeSeriesCV import TimeSeriesCV
cc = columns_contain


def make_grid():
    # return Bunch of hyperparameter settings
    return Bunch(
        # HP settings to test across all models
        n_months_back=(1, 2, 3, 6, 12),

        # HP settings to test for ElasticNet models
        alpha=(0.01, 0.03, 0.1, 0.3, 1.0),  # multiplies the penalty term
        l1_ratio=(0.0, 0.25, 0.50, 0.75, 1.0),  # 0 ==> L2 penalty, 1 ==> L1 penalty
        units_X=('natural', 'log'),
        units_y=('natural', 'log'),

        # HP settings to test for tree-based models
        # settings based on Anil Kocak's recommendations
        n_estimators=(10, 30, 100, 300),
        max_features=(1, 'log2', 'sqrt', 'auto'),
        max_depth=(1, 3, 10, 30, 100, 300),

        # HP setting to test for GradientBoostingRegression models
        learning_rate=(.10, .25, .50, .75, .99),
        # experiments demonstrated that the best loss is seldom quantile
        # loss_seq=('ls', 'quantile'),
        loss=('ls',),
    )


def make_control(argv):
    'return a Bunch'

    print argv
    parser = argparse.ArgumentParser()
    parser.add_argument('invocation')
    parser.add_argument(
        'features_hps_locality_month',
        nargs='?',
        default=None,
        type=arg_type.features_hps_locality_month,
        )
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--renameoutput', action='store_true')
    parser.add_argument('--makefile', nargs='*')
    arg = parser.parse_args(argv)
    arg.base_name = 'valavm'

    print arg

    dir_working = Path().dir_working()
    path_in_samples = dir_working + 'samples-train.csv'
    if arg.makefile is not None:
        if len(arg.makefile) == 0:
            arg.makefile = 'dell 16 roy 12 judith 7 hp 4'
        if len(arg.makefile) % 2 != 0:
            print 'must supply {system} {threads} ... pairs as argument to --makefile'
            os.exit(1)
        return Bunch(
            arg=arg,
            file_out_log='valavm-makefile',
            path_in_samples=path_in_samples,
            path_out_makefile=Path().dir_src() + 'valavm.makefile',
            path_out_src=Path().dir_src(),
            )
    if arg.renameoutput is not None:
        return Bunch(
            arg=arg,
            file_out_log='valavm-renameoutput',
            path_out_src=Path().dir_src(),
            )

    s = arg.features_hps_locality_month.split('-')
    assert len(s) == 4, s
    arg.features_group, arg.hps, arg.locality, arg.validation_month = s

    random_seed = 123
    random.seed(random_seed)

    dir_working = Path().dir_working()

    # assure output directory exists
    dir_path = '%svalavm/%s-%s-%s/' % (dir_working, arg.features_group, arg.hps, arg.locality)
    out_file_name = '%s.pickle' % arg.validation_month
    path_out_file = dir_path + out_file_name
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return Bunch(
        arg=arg,
        debug=False,
        dir_working=dir_working,
        file_out_log='valavm-%s' % arg.features_hps_locality_month,
        path_in_samples=dir_working + 'samples-train.csv',
        path_out_file=path_out_file,
        grid_seq=make_grid(),
        random_seed=random_seed,
        timer=Timer(),
    )


ResultKeyEn = collections.namedtuple(
    'ResultKeyEn',
    'n_months_back units_X units_y alpha l1_ratio',
)
ResultKeyGbr = collections.namedtuple(
    'ResultKeyGbr',
    'n_months_back n_estimators max_features max_depth loss learning_rate',
)
ResultKeyRfr = collections.namedtuple(
    'ResultKeyRfr',
    'n_months_back n_estimators max_features max_depth',
)
ResultValue = collections.namedtuple(
    'ResultValue',
    'actuals predictions',
)


class LocationSelector(object):
    def __init__(self, locality):
        locality_column_name = {
            'census': layout_transactions.census_tract,
            'city': layout_transactions.city,
            'global': None,
            'zip': layout_transactions.zip5,
            }[locality]
        self.locality_column_name = locality_column_name

    def location_values(self, df):
        'return values in the locality column'
        values = df[self.locality_column_name]
        return values

    def in_location(self, df, location):
        'return samples that are in the location'
        has_location = df[self.locality_column_name] == location
        subset = df.loc[has_location]
        assert sum(has_location) == len(subset)
        return subset


def fit_and_predict(samples, control, already_exists, save):
    'call save(ResultKey, ResultValue) for all the hps that do not exist'

    def make_result_keys(control):
        'return list of ResultKey'

        def en():
            'return list of ResultKenEn'
            result = []
            for n_months_back in control.grid_seq.n_months_back:
                for units_X in control.grid_seq.units_X:
                    for units_y in control.grid_seq.units_y:
                        for alpha in control.grid_seq.alpha:
                            for l1_ratio in control.grid_seq.l1_ratio:
                                item = ResultKeyEn(
                                    n_months_back=n_months_back,
                                    units_X=units_X,
                                    units_y=units_y,
                                    alpha=alpha,
                                    l1_ratio=l1_ratio,
                                    )
                                result.append(item)
            return result

        def gbr():
            'return list of ResultKeyGbr'
            result = []
            for n_months_back in control.grid_seq.n_months_back:
                for n_estimators in control.grid_seq.n_estimators:
                    for max_features in control.grid_seq.max_features:
                        for max_depth in control.grid_seq.max_depth:
                            for loss in control.grid_seq.loss:
                                for learning_rate in control.grid_seq.learning_rate:
                                    item = ResultKeyGbr(
                                        n_months_back=n_months_back,
                                        n_estimators=n_estimators,
                                        max_features=max_features,
                                        max_depth=max_depth,
                                        loss=loss,
                                        learning_rate=learning_rate,
                                        )
                                    result.append(item)
            return result

        def rfr():
            'return list of ResultKeyRfr'
            result = []
            for n_months_back in control.grid_seq.n_months_back:
                for n_estimators in control.grid_seq.n_estimators:
                    for max_features in control.grid_seq.max_features:
                        for max_depth in control.grid_seq.max_depth:
                            item = ResultKeyRfr(
                                n_months_back=n_months_back,
                                n_estimators=n_estimators,
                                max_features=max_features,
                                max_depth=max_depth,
                                )
                            result.append(item)
            return result

        hps = control.arg.hps
        result = []
        if hps == 'all' or hps == 'en':
            result.extend(en())
        if hps == 'all' or hps == 'gbr':
            result.extend(gbr())
        if hps == 'all' or hps == 'rfr':
            result.extend(rfr())
        return result

    def split_train_validate(n_months_back):
        '''return (train, validate)
        where
        - test contains only transactions in the validation_month
        - train contains only transactions in the n_months_back preceeding the
          validation_month
        '''
        validation_month = Month(control.arg.validation_month)
        ss = SampleSelector(samples)
        samples_validate = ss.in_month(validation_month)
        samples_train = ss.between_months(
            validation_month.decrement(n_months_back),
            validation_month.decrement(1),
            )
        return samples_train, samples_validate

    def make_model_name(result_key):
        if isinstance(result_key, ResultKeyEn):
            return 'ElasticNet'
        if isinstance(result_key, ResultKeyGbr):
            return 'GradientBoostingRegressor'
        if isinstance(result_key, ResultKeyRfr):
            return 'RandomForestRegressor'
        print 'unexpected result_key type', result_key, type(result_key)
        pdb.set_trace()

    def make_avm(result_key):
        'return avm using specified hyperparameters'
        model_name = make_model_name(result_key)
        if model_name == 'ElasticNet':
            return AVM.AVM(
                model_name=model_name,
                random_state=control.random_seed,
                units_X=result_key.units_X,
                units_y=result_key.units_y,
                alpha=result_key.alpha,
                l1_ratio=result_key.l1_ratio,
                features_group=control.arg.features_group,
                )
        elif model_name == 'GradientBoostingRegressor':
            return AVM.AVM(
                model_name=model_name,
                random_state=control.random_seed,
                learning_rate=result_key.learning_rate,
                loss=result_key.loss,
                alpha=0.5 if result_key.loss == 'quantile' else None,
                n_estimators=result_key.n_estimators,
                max_depth=result_key.max_depth,
                max_features=result_key.max_features,
                features_group=control.arg.features_group,
                )
        elif model_name == 'RandomForestRegressor':
            return AVM.AVM(
                model_name=model_name,
                random_state=control.random_seed,
                n_estimators=result_key.n_estimators,
                max_depth=result_key.max_depth,
                max_features=result_key.max_features,
                features_group=control.arg.features_group,
                )
        else:
            print 'bad result_key.model_name', result_key
            pdb.set_trace()

    def make_importances(model_name, fitted_avm):
        if model_name == 'ElasticNet':
            return {
                    'intercept': fitted_avm.intercept_,
                    'coefficients': fitted_avm.coef_,
                    'features_group': control.arg.features_group,
                    }
        else:
            # the tree-based models have the same structure for their important features
            return {
                    'feature_importances': fitted_avm.feature_importances_,
                    'features_group': control.arg.features_group,
                    }

    def define_fit_predict_importances(test=None, train=None, hp=None):
        'return (ResultValue, importances)'
        assert test is not None
        assert train is not None
        assert hp is not None
        pdb.set_trace()
        avm = make_avm(hp)
        fitted_avm = avm.fit(train)
        predictions = avm.predict(test)
        actuals = test[layout_transactions.price]
        return ResultValue(actuals, predictions), make_importances(avm.model_name, fitted_avm)

    def make_result_value(result_key=None, samples_train=None, samples_validate=None):
        'return ResultValue'
        avm = make_avm(result_key)
        fitted_avm = avm.fit(samples_train)
        predictions = avm.predict(samples_validate)
        actuals = samples_validate[layout_transactions.price]
        importances = make_importances(avm.model_name, fitted_avm)
        return ResultValue(actuals=actuals, predictions=predictions), importances

    location_selector = LocationSelector(control.arg.locality)
    result_keys = make_result_keys(control)
    for result_key_index, result_key in enumerate(result_keys):
        if already_exists(result_key):
            continue
        print 'result_key %d of %d' % (result_key_index + 1, len(result_keys))
        print result_key
        all_samples_train, all_samples_validate = split_train_validate(result_key.n_months_back)
        if control.arg.locality == 'global':
            # fit one model on all the training samples
            # use it to predict all the validation samples
            print 'global', result_key
            result_value, importances = make_result_value(
                result_key=result_key,
                samples_train=all_samples_train,
                samples_validate=all_samples_validate,
                )
            save(result_key, (result_value, importances))
        else:
            # fit one model for each location in the validation set (ex: city)
            # use it to predict just the validation samples in the same location
            locations = location_selector.location_values(all_samples_validate)
            unique_locations = set(locations)
            for location_index, location in enumerate(set(unique_locations)):
                print control.arg.locality, location
                location_samples_validate = location_selector.in_location(all_samples_validate, location)
                location_samples_train = location_selector.in_location(all_samples_train, location)
                print 'location %s (%d of %d): number of samples: %d training %d validation' % (
                    str(location),
                    location_index + 1,
                    len(unique_locations),
                    len(location_samples_validate),
                    len(location_samples_train),
                    )
                assert len(location_samples_validate) > 0, location_samples_validate
                if len(location_samples_train) == 0:
                    print 'no training samples for location', control.arg.locality, location, result_key
                    continue
                result_value, importances = make_result_value(
                    result_key=result_key,
                    samples_train=location_samples_train,
                    samples_validate=location_samples_validate,
                    )
                save(result_key, (result_value, importances))


FittedAvm = collections.namedtuple('FittedAVM', 'index key fitted')


def process_hps_all(control, samples):
    existing_keys_values = {}
    with open(control.path_out_file, 'rb') as prior:
        while True:
            try:
                record = pickle.load(prior)
                key, value = record
                existing_keys_values[key] = value
            except pickle.UnpicklingError as e:
                print key
                print e
                print 'ignored'
            except ValueError as e:
                print key
                print e
                print 'ignored'
            except EOFError:
                break
    print 'number of existing keys in output file:', len(existing_keys_values)
    control.timer.lap('read existing keys and values')

    # rewrite output file, staring with existing values
    with open(control.path_out_file, 'wb') as output:
        existing_keys = set(existing_keys_values.keys())

        def already_exists(key):
            return key in existing_keys

        def save(key, value):
            record = (key, value)
            pickle.dump(record, output)
            existing_keys.add(key)

        # write existing values
        for existing_key, existing_value in existing_keys_values.iteritems():
            save(existing_key, existing_value)
        if control.debug:
            print 'since debugging, did not re-write output file'
        control.timer.lap('wrote new output file with existings key and values')
        existing_keys_values = None

        # write new values
        fit_and_predict(samples, control, already_exists, save)
        control.timer.lap('create additional keys and values')


def renameoutput(control):
    'rename files, not directories'
    def make_initial_path():
        initial_path = '%s%s/' % (control.dir_working, control.arg.basename)
        return initial_path

    def make_dir_path(features_group, hps):
        return '%svalavm/%s-%s/' % (control.dir_working, features_group, hps)

    def make_new_file_path(features_group, hps, month):
        dir_path = make_dir_path(features_group, hps)
        file_name = '%s.pickle' % month
        file_path = dir_path + file_name
        return file_path

    def make_old_file_path(features_group, hps, month):
        dir_path = make_dir_path(features_group, hps)
        file_name = '%s-%s-%s.pickle' % (features_group, hps, month)
        file_path = dir_path + file_name
        return file_path

    months = (
        '200512',
        '200601', '200602', '200603', '200604', '200605', '200606',
        '200607', '200608', '200609', '200610', '200611', '200612',
        '200701', '200702', '200703', '200704', '200705', '200706',
        '200707', '200708', '200709', '200710', '200711', '200712',
        '200801', '200802', '200803', '200804', '200805', '200806',
        '200807', '200808', '200809', '200810', '200811', '200812',
        '200901', '200902')

    for features_group in ('s', 'sw', 'swp', 'swpn'):
        for hps in ('all',):
            for month in months:
                old_file_path = make_old_file_path(features_group, hps, month)
                new_file_path = make_new_file_path(features_group, hps, month)
                print 'rename', old_file_path, new_file_path
                os.rename(old_file_path, new_file_path)


def makefile(control):
    'write file valavm.makefile'
    months = (
        '200512',
        '200601', '200602', '200603', '200604', '200605', '200606',
        '200607', '200608', '200609', '200610', '200611', '200612',
        '200701', '200702', '200703', '200704', '200705', '200706',
        '200707', '200708', '200709', '200710', '200711', '200712',
        '200801', '200802', '200803', '200804', '200805', '200806',
        '200807', '200808', '200809', '200810', '200811', '200812',
        '200901', '200902')

    def make_jobs(args):
        jobs = {}
        for index in range(0, len(args), 2):
            system_name = args[index]
            hardware_threads = args[index + 1]
            jobs[system_name] = int(hardware_threads)
        return jobs

    def make_system_generator(jobs):
        systems = jobs.keys()
        for system in itertools.cycle(systems):
            for index in xrange(jobs[system]):
                yield system

    def append_lines(report, feature_group, hps, locality, month, system):
        system_lhs = 'valavm-%s-%s-%s-%s' % (feature_group, hps, locality, system)
        system_rhs = '../data/working/valavm/%s-%s-%s/%s.pickle' % (feature_group, hps, locality, month)
        system = '%s += %s' % (system_lhs, system_rhs)
        report.append(system)

        target_lhs = system_rhs
        target_rhs = 'valavm.py ' + control.path_in_samples
        target = '%s: %s' % (target_lhs, target_rhs)
        report.append(target)

        recipe = '\t~/anaconda2/bin/python valavm.py %s-%s-%s-%s' % (feature_group, hps, locality, month)
        report.append(recipe)

    def append_target(report, feature_group, hps, locality, system):
        thing = 'valavm-%s-%s-%s-%s' % (feature_group, hps, locality, system)
        line = '%s: $(%s)' % (thing, thing)
        report.append(line)

    args = control.arg.makefile.split(' ')
    jobs = make_jobs(args)
    # m = Makefile(jobs, control.path_in_samples)
    report = Report()
    report2 = Report()
    system_generator = make_system_generator(jobs)
    for feature_group in ('s', 'sw', 'swp', 'swpn'):
        for hps in ('all',):
            for locality in ('census', 'city', 'global'):
                for month in months:
                    system = system_generator.next()
                    append_lines(report, feature_group, hps, locality, month, system)
                    append_target(report, feature_group, hps, locality, system)
    for line in report2.iterlines():
        report.append(line)
    report.write(control.path_out_makefile)


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(base_name=control.file_out_log)
    print control

    if control.arg.renameoutput:
        renameoutput(control)
        sys.exit()

    if control.arg.makefile is not None:
        makefile(control)
        sys.exit()

    samples = pd.read_csv(
        control.path_in_samples,
        nrows=None if control.arg.test else None,
    )
    print 'samples.shape', samples.shape
    control.timer.lap('read samples')

    # assure output file exists
    if not os.path.exists(control.path_out_file):
        os.system('touch %s' % control.path_out_file)

    if control.arg.hps == 'all':
        process_hps_all(control, samples)
    else:
        print 'invalid arg.hps', control.arg.hps
        pdb.set_trace()

    print control
    if control.arg.test:
        print 'DISCARD OUTPUT: test'
    if control.debug:
        print 'DISCARD OUTPUT: debug'
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
