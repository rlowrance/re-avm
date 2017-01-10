'sweep hyparamaters over grid'

from __future__ import division

import pdb

import AVM
from columns_contain import columns_contain
import layout_transactions
import sweep_types
cc = columns_contain


def sweep_avm_hps(model_testperiod_grid, samples, random_state, just_test=False, verbose=True):
    'Return dictionary of test results for grid HPs on samples[test_period]'

    result = {}

    def vprint(s):
        if verbose:
            print s

    def max_features_s(max_features):
        'convert to 4-character string (for printing)'
        return max_features[:4] if isinstance(max_features, str) else ('%4.1f' % max_features)

    def fit_and_run(avm):
        'return a ResultValue'
        avm.fit(samples)
        mask = samples[layout_transactions.yyyymm] == test_period
        samples_yyyymm = samples[mask]
        predictions = avm.predict(samples_yyyymm)
        if predictions is None:
            print 'no predictions!'
            pdb.set_trace()
        actuals = samples_yyyymm[layout_transactions.price]
        return sweep_types.ResultValue(actuals, predictions)

    def search_en(n_months_back, test_period, grid):
        'search over ElasticNet HPs, appending to result'
        result = {}
        for units_X in grid.units_X_seq:
            for units_y in grid.units_y_seq:
                for alpha in grid.alpha_seq:
                    for l1_ratio in grid.l1_ratio_seq:
                        vprint(
                            '%6d %3s %1d %3s %3s %4.2f %4.2f' %
                            (test_period, 'en', n_months_back, units_X[:3], units_y[:3],
                             alpha, l1_ratio)
                        )
                        avm = AVM.AVM(
                            model_name='ElasticNet',
                            forecast_time_period=test_period,
                            random_state=random_state,
                            n_months_back=n_months_back,
                            units_X=units_X,
                            units_y=units_y,
                            alpha=alpha,
                            l1_ratio=l1_ratio,
                        )
                        result_key = sweep_types.ResultKeyEn(
                            n_months_back,
                            units_X,
                            units_y,
                            alpha,
                            l1_ratio,
                        )
                        pdb.set_trace()
                        result[result_key] = fit_and_run(avm)
                        if just_test:
                            return result
        return result

    def search_gbr(n_months_back, test_period, grid):
        'search over GradientBoostingRegressor HPs, appending to result'
        result = {}
        for n_estimators in grid.n_estimators_seq:
            for max_features in grid.max_features_seq:
                for max_depth in grid.max_depth_seq:
                    for loss in grid.loss_seq:
                        for learning_rate in grid.learning_rate_seq:
                            vprint(
                                '%6d %3s %1d %4d %4s %3d %8s %4.2f' %
                                (test_period, 'gbr', n_months_back,
                                 n_estimators, max_features_s(max_features), max_depth, loss, learning_rate)
                            )
                            avm = AVM.AVM(
                                model_name='GradientBoostingRegressor',
                                forecast_time_period=test_period,
                                random_state=random_state,
                                n_months_back=n_months_back,
                                learning_rate=learning_rate,
                                loss=loss,
                                alpha=.5 if loss == 'quantile' else None,
                                n_estimators=n_estimators,  # number of boosting stages
                                max_depth=max_depth,  # max depth of any tree
                                max_features=max_features,  # how many features to test when splitting
                            )
                            result_key = sweep_types.ResultKeyGbr(
                                n_months_back,
                                n_estimators,
                                max_features,
                                max_depth,
                                loss,
                                learning_rate,
                            )
                            pdb.set_trace()
                            result[result_key] = fit_and_run(avm)
                            if just_test:
                                return result
        pdb.set_trace()
        return result

    def search_rf(n_months_back, test_period, grid):
        'search over RandomForestRegressor HPs, appending to result'
        result = {}
        for n_estimators in grid.n_estimators_seq:
            for max_features in grid.max_features_seq:
                for max_depth in grid.max_depth_seq:
                    vprint(
                        '%6d %3s %1d %4d %4s %3d' %
                        (test_period, 'rfr', n_months_back,
                         n_estimators, max_features_s(max_features), max_depth)
                    )
                    avm = AVM.AVM(
                        model_name='RandomForestRegressor',
                        forecast_time_period=test_period,
                        random_state=random_state,
                        n_months_back=n_months_back,
                        n_estimators=n_estimators,  # number of boosting stages
                        max_depth=max_depth,  # max depth of any tree
                        max_features=max_features,  # how many features to test when splitting
                    )
                    result_key = sweep_types.ResultKeyRfr(
                        n_months_back,
                        n_estimators,
                        max_features,
                        max_depth,
                    )
                    pdb.set_trace()
                    result[result_key] = fit_and_run(avm)
                    if just_test:
                        return result
        return result

    # grid search for all model types
    pdb.set_trace()
    result = {}
    for model, testperiod_grid in model_testperiod_grid.iteritems():
        test_period, grid = testperiod_grid
        for n_months_back in grid.n_months_back_seq:
            if model == 'en':
                result = dict(result, **search_en(n_months_back, test_period, grid))
            if model == 'gb':
                more = search_gbr(n_months_back, test_period, grid)
                pdb.set_trace()
                result = dict(result, **more)
                result = dict(result, **search_gbr(n_months_back,test_period, grid))
            if model == 'rf':
               result = dict(result, search_rf(n_months_back, test_period, grid))
            if just_test:
                break

    return result
