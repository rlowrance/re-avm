'''create files contains estimated generalization errors for model

INPUT FILE
 WORKING/transactions-subset2.pickle

OUTPUT FILES
 WORKING/ege_week/YYYY-MM-DD/MODEL-TD/HP-FOLD.pickle  dict all_results
 WORKING/ege_month/YYYY-MM-DD/MODEL-TD/HP-FOLD.pickle  dict all_results
'''

import collections
import cPickle as pickle
import datetime
import numpy as np
import os
import pandas as pd
import pdb
from pprint import pprint
from sklearn import cross_validation
from sklearn import linear_model
from sklearn import ensemble
import sys
import warnings

from Bunch import Bunch
from DataframeAppender import DataframeAppender
from directory import directory
from Logger import Logger
import parse_command_line


def usage(msg=None):
    if msg is not None:
        print 'invocation error: ' + str(msg)
    print 'usage: python ege_week.py YYYY-MM-DD <other options>'
    print ' YYYY-MM-DD       mid-point of week; analyze -3 to +3 days'
    print ' --month          optional; test on next month, not next week'
    print ' --model {lr|rf}  which model to run'
    print ' --td <range>     training_days'
    print ' --hpd <range>    required iff model is rf; max_depths to model'
    print ' --hpw <range>    required iff model is rf; weight functions to model'
    print ' --hpx <form>     required iff mode is lr; transformation to x'
    print ' --hpy <form>     required iff mode is lr; transformation to y'
    print ' --test           optional; if present, program runs in test mode'
    print 'where'
    print ' <form>  is {lin|log}+ saying whether the variable is in natural or log units'
    print ' <range> is start [stop [step]], just like Python\'s range(start,stop,step)'
    sys.exit(1)


DateRange = collections.namedtuple('DateRange', 'first last')


def make_DateRange(mid, half_range):
    return DateRange(first=mid - datetime.timedelta(half_range),
                     last=mid + datetime.timedelta(half_range),
                     )


def make_predictors():
    '''return dict key: column name, value: whether and how to convert to log domain

    Include only features of the census and tax roll, not the assessment,
    because previous work found that using features derived from the
    assessment degraded estimated generalization errors.

    NOTE: the columns in the x_array objects passed to scikit learn are in
    this order. FIXME: that must be wrong, as we return a dictionary
    '''
    # earlier version returned a dictionary, which invalided the assumption
    # about column order in x
    result = (  # the columns in the x_arrays are in this order
        ('fraction.owner.occupied', None),
        ('FIREPLACE.NUMBER', 'log1p'),
        ('BEDROOMS', 'log1p'),
        ('BASEMENT.SQUARE.FEET', 'log1p'),
        ('LAND.SQUARE.FOOTAGE', 'log'),
        ('zip5.has.industry', None),
        ('census.tract.has.industry', None),
        ('census.tract.has.park', None),
        ('STORIES.NUMBER', 'log1p'),
        ('census.tract.has.school', None),
        ('TOTAL.BATHS.CALCULATED', 'log1p'),
        ('median.household.income', 'log'),  # not log feature in earlier version
        ('LIVING.SQUARE.FEET', 'log'),
        ('has.pool', None),
        ('zip5.has.retail', None),
        ('census.tract.has.retail', None),
        ('is.new.construction', None),
        ('avg.commute', None),
        ('zip5.has.park', None),
        ('PARKING.SPACES', 'log1p'),
        ('zip5.has.school', None),
        ('TOTAL.ROOMS', 'log1p'),
        ('age', None),
        ('age2', None),
        ('effective.age', None),
        ('effective.age2', None),
    )
    return result


class CensusAdjacencies(object):
    def __init__(self):
        path = directory('working') + 'census_tract_adjacent.pickle'
        f = open(path, 'rb')
        self.adjacent = pickle.load(f)
        f.close()

    def adjacen(self, census_tract):
        return self.adjacent.get(census_tract, None)


def make_control(argv):
    'Return control Bunch'''

    print 'argv'
    pprint(argv)

    if len(argv) < 3:
        usage('missing invocation options')

    def make_sale_date(s):
        year, month, day = s.split('-')
        return datetime.date(int(year), int(month), int(day))

    pcl = parse_command_line.ParseCommandLine(argv)
    arg = Bunch(
        base_name=argv[0].split('.')[0],
        hpd=pcl.get_range('--hpd') if pcl.has_arg('--hpd') else None,
        hpw=pcl.get_range('--hpw') if pcl.has_arg('--hpw') else None,
        hpx=pcl.get_arg('--hpx') if pcl.has_arg('--hpx') else None,
        hpy=pcl.get_arg('--hpy') if pcl.has_arg('--hpy') else None,
        model=pcl.get_arg('--model'),
        month=pcl.has_arg('--month'),
        sale_date=make_sale_date(argv[1]),
        td=pcl.get_range('--td'),
        test=pcl.has_arg('--test'),
    )
    print 'arg'
    print arg
    # check for missing options
    if arg.model is None:
        usage('missing --model')
    if arg.td is None:
        usage('missing --td')

    # validate combinations of invocation options
    if arg.model == 'lr':
        if arg.hpx is None or arg.hpy is None:
            usage('model lr requires --hpx and --hpy')
    elif arg.model == 'rf':
        if arg.hpd is None or arg.hpw is None:
            usage('model rf requires --hpd and --hpw')
    else:
        usage('bad --model: %s' % str(arg.model))

    random_seed = 123
    now = datetime.datetime.now()
    predictors = make_predictors()
    print 'number of predictors', len(predictors)
    sale_date_range = make_DateRange(arg.sale_date, 15 if arg.month else 3)
    log_file_name = arg.base_name + '.' + now.isoformat('T') + '.log'
    # dir_out: WORKING/ege_[month|week]/<sale_date>/
    dir_out = (directory('working') +
               'ege_' +
               ('month' if arg.month else 'week') +
               '/' + argv[1] + '/'
               )

    debug = False
    test = arg.test

    b = Bunch(
        arg=arg,
        census_adjacencies=CensusAdjacencies(),
        date_column='python.sale_date',
        debug=debug,
        dir_out=dir_out,
        n_folds=2 if test else 10,
        n_rf_estimators=100 if test else 1000,  # num trees in a random forest
        path_in_old=directory('working') + 'transactions-subset2.pickle',
        path_in=directory('working') + 'transactions-subset3-subset-train.csv',
        path_log=directory('log') + log_file_name,
        predictors=predictors,
        price_column='SALE.AMOUNT',
        random_seed=random_seed,
        relevant_date_range=DateRange(first=datetime.date(2003, 1, 1), last=datetime.date(2009, 3, 31)),
        sale_date_range=sale_date_range,
        start_time=now,
        test=test,
        use_old_input=True,
    )
    return b


def elapsed_time(start_time):
    return datetime.datetime.now() - start_time


def x(mode, df, predictors):
    '''return 2D np.array, with df x values possibly transformed to log

    RETURNS array: np.array 2D
    '''
    def transform(v, mode, transformation):
        if mode is None:
            return v
        if mode == 'linear' or mode == 'lin':
            return v
        if mode == 'log':
            if transformation is None:
                return v
            if transformation == 'log':
                return np.log(v)
            if transformation == 'log1p':
                return np.log1p(v)
            raise RuntimeError('bad transformation: ' + str(transformation))
        raise RuntimeError('bad mode:' + str(mode))

    array = np.empty(shape=(df.shape[0], len(predictors)),
                     dtype=np.float64).T
    # build up in transposed form
    index = 0
    for predictor_name, transformation in predictors:
        v = transform(df[predictor_name].values, mode, transformation)
        array[index] = v
        index += 1
    return array.T


def y(mode, df, price_column):
    '''return np.array 1D with transformed price column from df'''
    df2 = df.copy(deep=True)
    if mode == 'log':
        df2[price_column] = pd.Series(np.log(df[price_column]), index=df.index)
    array = np.array(df2[price_column].as_matrix(), np.float64)
    return array


def mask_in_date_range(df, date_range):
    df_date = df['sale.python_date']
    return (df_date >= date_range.first) & (df_date <= date_range.last)


def samples_in_date_range(df, date_range):
    'return new df'
    return df[mask_in_date_range(df, date_range)]


def add_age(df, sale_date):
    'Return new df with extra columns for age and effective age'
    column_names = df.columns.tolist()
    if 'age' in column_names:
        print column_names
        print 'age in column_names'
        pdb.set_trace()
    assert('age' not in column_names)
    assert('age2' not in column_names)
    assert('effective.age' not in column_names)
    assert('effective.age2' not in column_names)

    sale_year = df['sale.year']

    def age(column_name):
        'age from sale_date to specified column'
        age_in_years = sale_year - df[column_name].values
        return pd.Series(age_in_years, index=df.index)

    result = df.copy(deep=True)

    result['age'] = age('YEAR.BUILT')
    result['effective.age'] = age('EFFECTIVE.YEAR.BUILT')
    result['age2'] = result['age'] * result['age']
    result['effective.age2'] = result['effective.age'] * result['effective.age']

    return result


def squeeze(obj, verbose=False):
    'replace np.array float64 with np.array float32'
    if isinstance(obj, dict):
        return {k: squeeze(v) for k, v in obj.iteritems()}
    if isinstance(obj, np.ndarray) and obj.dtype == np.float64:
        return np.array(obj, dtype=np.float32)
    return obj


def make_weights(query, train_df, hpw, control):
    'return numpy.array of weights for each sample'
    if hpw == 1:
        return np.ones(len(train_df))
    else:
        print 'bad hpw: %s' % hpw


def sweep_hp_lr(train_df, validate_df, control):
    'sweep hyperparameters, fitting and predicting for each combination'
    def x_matrix(df, transform):
        augmented = add_age(df, control.arg.sale_date)
        return x(transform, augmented, control.predictors)

    def y_vector(df, transform):
        return y(transform, df, control.price_column)

    verbose = True
    LR = linear_model.LinearRegression
    results = {}
    for hpx in control.arg.hpx:
        for hpy in control.arg.hpy:
            if verbose:
                print 'sweep_hr_lr hpx %s hpy %s' % (hpx, hpy)
            model = LR(fit_intercept=True,
                       normalize=True,
                       copy_X=False,
                       )
            train_x = x_matrix(train_df, hpx)
            train_y = y_vector(train_df, hpy)
            model.fit(train_x, train_y)
            estimates = model.predict(x_matrix(validate_df, hpx))
            actuals = y_vector(validate_df, hpy)
            attributes = {
                'coef_': model.coef_,
                'intercept_': model.intercept_
            }
            results[('y_transform', hpy), ('x_transform', hpx)] = squeeze({
                'estimate': estimates,
                'actual': actuals,
                'attributes': attributes
            })
    return results


def sweep_hp_rf(train_df, validate_df, control):
    'fit a model and validate a model for each hyperparameter'
    def x_matrix(df):
        augmented = add_age(df, control.arg.sale_date)
        return x(None, augmented, control.predictors)

    def y_vector(df):
        return y(None, df, control.price_column)

    verbose = True
    RFR = ensemble.RandomForestRegressor
    train_x = x_matrix(train_df)
    train_y = y_vector(train_df)
    results = {}
    for hpd in control.arg.hpd:
        for hpw in control.arg.hpw:
            for validate_row_index in xrange(len(validate_df)):
                if verbose:
                    print 'sweep_hp_rf hpd %d hpw %d validate_row_index %d of %d' % (
                        hpd, hpw, validate_row_index, len(validate_df))
                validate_row = validate_df[validate_row_index: validate_row_index + 1]
                model = RFR(n_estimators=control.n_rf_estimators,  # number of trees
                            random_state=control.random_seed,
                            max_depth=hpd)
                weights = make_weights(validate_row, train_df, hpw, control)
                model.fit(train_x, train_y, weights)
                estimate = squeeze(model.predict(x_matrix(validate_row))[0])
                actual = squeeze(y_vector(validate_row)[0])
                # Don't keep some attributes
                #  oob attributes are not produced because we didn't ask for them
                #  estimators_ contains a fitted model for each estimate
                attributes = {
                    'feature_importances_': model.feature_importances_,
                }
                results[('max_depth', hpd), ('weight_scheme_index', hpw)] = squeeze({
                    'estimate': estimate,
                    'actual': actual,
                    'attributes': attributes,
                })
    return results


def cross_validate(df, control):
    'produce estimated generalization errors'
    verbose = True
    results = {}
    fold_number = -1
    sale_dates_mask = mask_in_date_range(df, control.sale_date_range)
    skf = cross_validation.StratifiedKFold(sale_dates_mask, control.n_folds)
    for train_indices, validate_indices in skf:
        fold_number += 1
        fold_train_all = df.iloc[train_indices].copy(deep=True)
        fold_validate_all = df.iloc[validate_indices].copy(deep=True)
        for td in control.arg.td:
            if verbose:
                print 'cross_validate fold %d of %d training_days %d' % (
                    fold_number, control.n_folds, td)
            fold_train = samples_in_date_range(
                fold_train_all,
                DateRange(first=control.arg.sale_date - datetime.timedelta(td),
                          last=control.arg.sale_date - datetime.timedelta(1))
            )
            fold_validate = samples_in_date_range(
                fold_validate_all,
                control.sale_date_range
            )
            if control.arg.model == 'lr':
                d = sweep_hp_lr(fold_train, fold_validate, control)
            elif control.arg.model == 'rf':
                d = sweep_hp_rf(fold_train, fold_validate, control)
                # d = cross_validate_rf(fold_train, fold_validate, control)
            else:
                print 'bad model: %s' % control.model
                pdb.set_trace()
            results[(('fn', fold_number), ('td', td))] = d
    return results


def predict_next(df, control):
    'fit each model and predict transaction in next period'
    verbose = True
    for td in control.arg.td:
        if verbose:
            print 'predict_next training_days %d' % td
        last_sale_date = control.sale_date_range.last
        train_df = samples_in_date_range(
            df,
            DateRange(first=last_sale_date - datetime.timedelta(td),
                      last=last_sale_date)
        )
        next_days = 30 if control.arg.month else 7
        test_df = samples_in_date_range(
            df,
            DateRange(first=last_sale_date,
                      last=last_sale_date + datetime.timedelta(next_days))
        )
        if control.arg.model == 'lr':
            return sweep_hp_lr(train_df, test_df, control)
        elif control.arg.model == 'rf':
            return sweep_hp_rf(train_df, test_df, control)
        else:
            print 'bad model: %s' % control.arg.model


def fit_and_test_models(df_all, control):
    'Return all_results dict'
    # throw away irrelevant transactions
    df_relevant = samples_in_date_range(df_all, control.relevant_date_range)

    results_cv = cross_validate(df_relevant, control)
    results_next = predict_next(df_relevant, control)

    pdb.set_trace()
    return results_cv, results_next


def main(argv):
    #  warnings.filterwarnings('error')  # convert warnings to errors
    control = make_control(argv)

    sys.stdout = Logger(logfile_path=control.path_log)  # print also write to log file
    print control

    # read input
    if control.use_old_input:
        f = open(control.path_in_old, 'rb')
        df_loaded = pickle.load(f)
        f.close()
    else:
        df_loaded = pd.read_csv(control.path_in, engine='c')

    df_loaded_copy = df_loaded.copy(deep=True)  # make sure df_loaded isn't changed

    results_cv, results_next = fit_and_test_models(df_loaded, control)
    assert(df_loaded.equals(df_loaded_copy))

    # write results
    def file_name(key):
        'model-foldNumber-trainingDays'
        assert len(key) == 2, key
        s = '%s-%s-%s' % (control.arg.model, key[0], key[1])
        return s

    def write(dir_prefix, results):
        for k, v in results.iteritems():
            directory = control.dir_out + dir_prefix
            if not os.path.exists(directory):
                os.makedirs(directory)
            f = open(directory + file_name(k), 'wb')
            pickle.dump((k, v), f)
            f.close()

    write('cv/', results_cv)
    write('next/', results_next)

    print 'ok'


if __name__ == "__main__":
    if False:
        # quite pyflakes warnings
        pdb.set_trace()
        pprint(None)
        np.all()
        pd.Series()
    main(sys.argv)
