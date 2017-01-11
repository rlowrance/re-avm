'''fit many models on the real estate data

INVOCATION
  python fit.py DATA MODEL FOCUS LASTMONTH
where
 DATA         in {train, all} specifies which data in Working/samples2 to use
 MODEL        in {en, gb, rf} specified which model to use
 LASTMONTH    like YYYYMM specfies the last month of training data
 NEIGHBORHOOD in {all, city_name} specifies whether to train a model on all cities or just the specified city

INPUTS
 WORKING/samples2/train.csv or
 WORKING/samples2/all.csv

OUTPUTS
 WORKING/fit[-item]/DATA-MODEL-LASTMONTH/NEIGHBORHOOD/<filename>.pickle
where 
 <filename> is the hyperparameters for the model
 the pickle files contains either
  (True, <fitted-model>)
  (False, <error message explaining why model could not be fitted)
'''

from __future__ import division

import argparse
import collections
import cPickle as pickle
import datetime
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
import dirutility
# from Features import Features
from HPs import HPs
import layout_transactions
from Logger import Logger
from Month import Month
from Path import Path
from Report import Report
from SampleSelector import SampleSelector
from valavmtypes import ResultKeyEn, ResultKeyGbr, ResultKeyRfr, ResultValue
from Timer import Timer
# from TimeSeriesCV import TimeSeriesCV
cc = columns_contain


def make_grid():
    'return all hyperparameter grid points'
    return Bunch(
        # HP settings to test across all models
        # n_months_back=(1, 2, 3, 6, 12),  FOR VALAVM
        n_months_back=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 24),

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
    parser.add_argument('data', choices=['all', 'train'])
    parser.add_argument('model', choices=['en', 'gb', 'rf'])
    parser.add_argument('last_month')
    parser.add_argument('neighborhood')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trace', action='store_true')
    arg = parser.parse_args(argv)
    arg.me = arg.invocation.split('.')[0]

    if arg.trace:
        pdb.set_trace()

    arg.last = Month(arg.last_month)  # convert to Month and validate value

    # convert arg.neighborhood into arg.all and arg.city
    arg.city = (
        None if arg.neighborhood == 'all' else
        arg.neighborhood.replace('_', ' ')
    )

    random_seed = 123
    random.seed(random_seed)

    dir_working = Path().dir_working()
    fit_dir = (
        os.path.join(dir_working, arg.me + '-test') if arg.test else
        os.path.join(dir_working, arg.me)
    )
    last_dir = '%s-%s-%s-%s' % (arg.data, arg.model, arg.last_month, arg.neighborhood)
    path_out_dir = os.path.join(fit_dir, last_dir, '')
    dirutility.assure_exists(path_out_dir)

    return Bunch(
        arg=arg,
        grid=make_grid(),
        path_in_dir=os.path.join(dir_working, 'samples2', ''),
        path_out_dir=path_out_dir,
        path_out_log=os.path.join(path_out_dir, '0log.txt'),
        random_seed=random_seed,
        timer=Timer(),
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


def split_train_validate(n_months_back, samples, validation_month):
    '''return (train, validate)
    where
    - test contains only transactions in the validation_month
    - train contains only transactions in the n_months_back preceeding the
      validation_month
    '''
    the_validation_month = Month(validation_month)
    ss = SampleSelector(samples)
    samples_validate = ss.in_month(the_validation_month)
    samples_train = ss.between_months(
        the_validation_month.decrement(n_months_back),
        the_validation_month.decrement(1),
        )
    return samples_train, samples_validate


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


def make_result_value(
        control=None,
        result_key=None,
        samples_train=None,
        samples_validate=None,
        features_group=None):
    'return (ResultValue, importances)'
    assert control is not None
    assert result_key is not None
    assert samples_train is not None
    assert samples_validate is not None
    assert features_group is not None

    def make_avm(result_key):
        'return avm using specified hyperparameters'
        model_name = (
            'ElasticNet' if isinstance(result_key, ResultKeyEn) else
            'GradientBoostingRegressor' if isinstance(result_key, ResultKeyGbr) else
            'RandomForestRegressor' if isinstance(result_key, ResultKeyRfr) else
            None)
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
            print 'bad model_name', (model_name, result_key)
            pdb.set_trace()

    def make_importances(model_name, fitted_avm):
        if model_name == 'ElasticNet':
            return {
                    'intercept': fitted_avm.intercept_,
                    'coefficients': fitted_avm.coef_,
                    'features_group': features_group,
                    }
        else:
            # the tree-based models have the same structure for their important features
            return {
                    'feature_importances': fitted_avm.feature_importances_,
                    'features_group': features_group,
                    }

    avm = make_avm(result_key)
    fitted_avm = avm.fit(samples_train)
    predictions = avm.predict(samples_validate)
    actuals = samples_validate[layout_transactions.price]
    importances = make_importances(avm.model_name, fitted_avm)
    return ResultValue(actuals=actuals, predictions=predictions), importances


def fit_and_predict(samples, control, already_exists, save):
    'call save(ResultKey, ResultValue) for all the hps that do not exist in the output file'

    pdb.set_trace()
    location_selector = LocationSelector(control.arg.locality)
    result_keys = make_result_keys(control)
    for result_key_index, result_key in enumerate(result_keys):
        all_samples_train, all_samples_validate = split_train_validate(
            result_key.n_months_back,
            samples,
            control.arg.validation_month,
            )
        if control.arg.locality == 'global':
            if already_exists(result_key):
                continue
            print 'result_key %d of %d' % (result_key_index + 1, len(result_keys))
            print result_key
            # fit one model on all the training samples
            # use it to predict all the validation samples
            print 'global', result_key
            result_value, importances = make_result_value(
                result_key=result_key,
                samples_train=all_samples_train,
                samples_validate=all_samples_validate,
                features_group=control.arg.features_group,
                )
            save(result_key, (result_value, importances))
        else:
            # fit one model for each location in the validation set (ex: city)
            # use it to predict just the validation samples in the same location
            pdb.set_trace()
            locations = location_selector.location_values(all_samples_validate)
            unique_locations = set(locations)
            for location_index, location in enumerate(set(unique_locations)):
                print control.arg.locality, location
                if already_exists(result_key, location):
                    continue
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
                    features_group=control.arg.features_group,
                    )
                save(result_key, (result_value, importances), location)


FittedAvm = collections.namedtuple('FittedAVM', 'index key fitted')


def read_existing_keys_values(path, timer):
    'return dict of keys and values found in file at path'
    existing_keys_values = {}
    with open(path, 'rb') as prior:
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
    timer.lap('read existing keys and values')
    return existing_keys_values


def process_hps_all_global(control, samples):
    'append new keys and values to the known global output file'
    # rewrite output file, staring with existing values
    assert control.arg.locality == 'global'
    existing_keys_values = read_existing_keys_values(control.path_out_file, control.timer)

    with open(control.path_out_file, 'wb') as output:
        written_keys = set()

        # rewrite existing values
        existing_keys_values = read_existing_keys_values(control.path_out_file, control.timer)
        count = 0
        for existing_key, existing_value in existing_keys_values.iteritems():
            record = (existing_key, existing_value)
            pickle.dump(record, output)
            written_keys.add(existing_key)
            count += 1
        control.timer.lap('rewrote new output file with %d existing keys and valuess' % count)

        # create and write new values
        for result_key in make_result_keys(control):
            if result_key in written_keys:
                continue
            pdb.set_trace()
            train, validate = split_train_validate(
                result_key.n_months_back,
                samples,
                control.arg.validation_month,
                )
            result_value, importances = make_result_value(
                result_key=result_key,
                samples_train=train,
                samples_validate=validate,
                features_group=control.arg.features_group,
                control=control,
                )
            record = (result_key, (result_value, importances))
            pickle.dump(record, output)
            control.timer.lap('create one additional key and value')
        control.timer.lap('create all additional keys and values')


class KeyTracker(object):
    def __init__(self, path_out_file_template, timer):
        self.path_out_file_template = path_out_file_template
        self.timer = timer

        self.existing_keys = {}
        self.last_location = None
        self.first_already_existing_call = True

    def add(self, key):
        pdb.set_trace()
        self.existing.add(key)

    def already_exists(self, key, location):
        pdb.set_trace()
        if self.last_locations is None or location != self.last_location:
            pass
            # create new file
            # path_out_file = self.path_out_file_template % location
            # existings_keys_values = read_existing_keys_values(path_out_file, self.timer)
            # rewrite the output file
            # with open(path_out_file, 'wb') as output:
            # pass
#
#                for existing_key, existing_value in existing_keys_values.iteritems():
#                    self.save(

        if self.first_already_exists_call:
            self.location = location
            self.make_existing_keys(location)
            self.first_already_exists_call = False
        assert location == self.location, 'location changed'
        return key in self.existing_keys

    def make_existing_keys(self, location):
        pdb.set_trace()
        path = self.path_out_file % location
        existings_keys_values = read_existing_keys_values(path)
        self.existing_keys = existings_keys_values.keys()

    def save(self, key, value, location):
        pdb.set_trace()
        assert location == self.location, 'location changed'
        pass


def process_hps_all_local(control, samples):
    'append new keys and values to the output files corresponding to the locations'
    # we don't know the output file name
    # there is a different output file name for every location handed to save()

    def append_to_location_file(location, location_selector):
        'append new keys and values to the location file'
        path = control.path_out_file % location
        with open(path, 'wb') as output:
            written_keys = set()

            # rewrite existing values
            existing_keys_values = read_existing_keys_values(path, control.timer)
            count = 0
            for existing_key, existing_value in existing_keys_values.iteritems():
                record = (existing_key, existing_value)
                pickle.dump(record, output)
                written_keys.add(existing_key)
                count += 1
            control.timer.lap('rewrote new output file with existing keys and values')

            # create and write new values
            for result_key in make_result_keys(control):
                if result_key in written_keys:
                    continue
                in_location_samples = location_selector.in_location(samples, location)
                if len(in_location_samples) == 0:
                    print 'skipping %s, as no samples for that location' % location
                    continue
                train, validate = split_train_validate(
                    result_key.n_months_back,
                    in_location_samples,
                    control.arg.validation_month,
                    )
                if len(train) == 0:
                    print 'skipping %s, as no training data' % location
                    continue
                if len(validate) == 0:
                    print 'skippig %s, as no validation data' % location
                    continue
                print 'location %s samples %d validation_month %s n_months_back %d train %d validation %d' % (
                    location,
                    len(in_location_samples),
                    control.arg.validation_month,
                    result_key.n_months_back,
                    len(train),
                    len(validate),
                    )
                result_value, importances = make_result_value(
                    result_key=result_key,
                    samples_train=train,
                    samples_validate=validate,
                    features_group=control.arg.features_group,
                    control=control,
                    )
                record = (result_key, (result_value, importances))
                pickle.dump(record, output)
                control.timer.lap('create on additional key value in location %s' % location)
            control.timer.lap('create additional keys and values')

    location_selector = LocationSelector(control.arg.locality)
    locations = location_selector.location_values(samples)
    unique_locations = set(locations)
    print 'found %d unique locations' % len(unique_locations)
    for location in unique_locations:
        append_to_location_file(location, location_selector)


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
    '''write file valavm.makefile with these targets
    valavm-{feature_group}-{locality}-{system}
    all
    '''
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

    def make_system_months(jobs, months):
        result = collections.defaultdict(list)
        system_generator = make_system_generator(jobs)
        for month in months:
            system = system_generator.next()
            result[system].append(month)
        return result

    def make_variable(feature_group, locality, system, month):
        lhs = 'valavm-%s-all-%s-%s' % (feature_group, locality, system)
        rhs = '../data/working/valavm/%s-all-%s/%s' % (feature_group, locality, month)
        line = '%s += %s' % (lhs, rhs)
        return line

    def make_target(feature_group, locality, system):
        var = 'valavm-%s-all-%s-%s' % (feature_group, locality, system)
        line = '%s : $(%s)' % (var, var)
        return line

    def make_rule(feature_group, locality, month):
        target_lhs = '../data/working/valavm/%s-all-%s/%s' % (feature_group, locality, month)
        target_rhs = 'valavm.py ' + control.path_in_samples
        target_line = '%s : %s' % (target_lhs, target_rhs)

        recipe_line = '\t~/anaconda2/bin/python valavm.py %s-all-%s-%s' % (feature_group, locality, month)

        return (target_line, recipe_line)

    args = control.arg.makefile.split(' ')
    jobs = make_jobs(args)
    system_months = make_system_months(jobs, months)

    report_variables = Report()
    report_variables.append('# valavm variables')
    report_targets = Report()
    report_targets.append('# valavm targets')
    report_rules = Report()
    report_rules.append('# valavm rules')
    # for now, only implement hps 'all'
    for feature_group in ('s', 'sw', 'swp', 'swpn'):
        for locality in ('city', 'global'):
            for system in jobs.keys():
                for month in system_months[system]:
                    report_variables.append(make_variable(feature_group, locality, system, month))
                    report_rules.append_lines(make_rule(feature_group, locality, month))
                report_targets.append(make_target(feature_group, locality, system))
    report_variables.append_report(report_targets)
    report_variables.append_report(report_rules)
    report_variables.write(control.path_out_makefile)
    return


def mainOLD(argv):
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
        if control.arg.locality == 'global':
            process_hps_all_global(control, samples)
        else:
            process_hps_all_local(control, samples)
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


def select_in_time_period(df, last_month_str, n_months_back):
    'return subset of DataFrame df that are in the time period'

    first_date_float = float(Month(last_month_str).decrement(n_months_back).as_int() * 100 + 1)
    next_month = Month(last_month_str).increment()
    last_date = datetime.date(next_month.year, next_month.month, 1) - datetime.timedelta(1)
    last_date_float = last_date.year * 10000.0 + last_date.month * 100.0 + last_date.day

    sale_date_column = layout_transactions.sale_date
    sale_dates_float = df[sale_date_column]  # type is float
    assert isinstance(sale_dates_float.iloc[0], float)

    mask1 = sale_dates_float >= first_date_float
    mask2 = sale_dates_float <= last_date_float
    mask_in_range = mask1 & mask2
    df_in_range = df.loc[mask_in_range]
    return df_in_range


def select_in_city(df, city):
    'return subset of DataFrame df that are in the city'
    cities = df[layout_transactions.city]
    mask = cities == city
    df_in_city = df.loc[mask]
    return df_in_city


def do_work(control):
    path_in = os.path.join(control.path_in_dir, control.arg.data + '.csv')
    training_data = pd.read_csv(
        path_in,
        nrows=10 if control.arg.test else None,
        usecols=None,  # TODO: change to columns we actually use
        low_memory=False,
    )
    print 'read %d rows of training data from file %s' % (len(training_data), path_in)
    pdb.set_trace()
    for hps in HPs.iter_hps_model(control.arg.model):
        print hps
        # reduce training data 
        in_time_period = select_in_time_period(
            training_data,
            control.arg.last_month,
            hps['n_months_back'],
        )
        in_neighborhood = (
            in_time_period if control.arg.neighborhood == 'all' else
            select_in_city(in_time_period, control.arg.city)
        )
        print '#training samples: %d, of which in time period: %d' % (len(training_data), len(in_time_period))
        print '#in time period: %d, of which in neighborhood: %d' % (len(in_time_period), len(in_neighborhood))
        pdb.set_trace()
        if len(in_neighborhood) == 0:
            print 'skipping fitting of model, because no training samples for those hyperparameters'
            continue
        if control.arg.model == 'en':
            fit_en(control, in_neighborhood, hps)
        elif control.arg.model == 'gb':
            fit_gb(control, in_neighborhood, hps)
        else:
            fit_rf(control, in_nieghborhood, hps)

    assert False, 'write more'


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(control.path_out_log)  # now print statements also write to the log file
    print control
    lap = control.timer.lap

    do_work(control)

    lap('work completed')
    if control.test:
        print 'DISCARD OUTPUT: test'
    print control
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
