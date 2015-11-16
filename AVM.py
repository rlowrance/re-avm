'Automated Valuation Model'

import pdb
import numpy as np
import pandas as pd
from pprint import pprint
import sklearn
import sklearn.ensemble
import sklearn.linear_model


from columns_contain import columns_contain
from Features import Features
import layout_transactions
cc = columns_contain


def avm_scoring(estimator, df):
    'return error from using fitted estimator with test data in the dataframe'
    # TODO: make a static method of class AVM
    assert isinstance(estimator, AVM)
    X, y = estimator.extract_and_transform(df)
    assert len(y) > 0
    y_hat = estimator.predict(df)
    errors = y_hat - y
    median_abs_error = np.median(np.abs(errors))
    return -median_abs_error  # because GridSearchCV chooses the model with the score


class AVM(sklearn.base.BaseEstimator):
    'one estimator for two underlying models: ElasticNet and RandomForestRegressor'
    def __init__(self,
                 model_name=None,          # parameters for all models
                 forecast_time_period=None,
                 n_months_back=None,
                 random_state=None,
                 verbose=0,
                 alpha=None,               # for ElasticNet
                 l1_ratio=None,
                 units_X=None,
                 units_y=None,
                 n_estimators=None,        # for RandomForestRegressor
                 max_depth=None,
                 max_features=None,
                 ):
        # NOTE: just capture the parameters (to conform to the sklearn protocol)
        self.model_name = model_name
        self.forecast_time_period = forecast_time_period
        self.n_months_back = n_months_back
        self.random_state = random_state
        self.verbose = verbose

        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.units_X = units_X
        self.units_y = units_y

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features

    def fit(self, df):
        'construct and fit df that contains X and y'
        def fit_elastic_net(X_train, y_train):
            if self.verbose > 0:
                print 'fit elastic net: %s~%s alpha: %f l1_ratio: %f' % (
                    self.units_X, self.units_y, self.alpha, self.l1_ratio)

            assert self.alpha > 0.0, self.l1_ratio  # otherwise, not reliable
            self.model = sklearn.linear_model.ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                fit_intercept=True,
                normalize=True,
                selection='random',
                random_state=self.random_state,
            )
            self.model.fit(X_train, y_train)
            return self

        def fit_random_forest_regressor(X_train, y_train):
            if self.verbose > 0:
                print (
                    'fit random forest regressor',
                    self.forecast_time_period,
                    self.n_estimators,
                    self.max_depth,
                    self.max_features,
                    self.n_months_back,
                )
            self.model = sklearn.ensemble.RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                max_features=self.max_features,
            )
            self.model.fit(X_train, y_train)
            return self

        mask = df[layout_transactions.yyyymm] < self.forecast_time_period
        df_period = df[mask]
        kept_yyyymm = []
        last_kept_yyyymm = self.forecast_time_period - 1
        for n in xrange(self.n_months_back):
            if last_kept_yyyymm % 100 == 0:
                # ex: found 200900; need to convert to 200812
                last_kept_yyyymm += - 100 + 12  # adjust year and month
            kept_yyyymm.append(last_kept_yyyymm)
            last_kept_yyyymm -= 1
        mask_kept = df_period[layout_transactions.yyyymm].isin(kept_yyyymm)
        df_kept = df_period[mask_kept]
        if self.verbose > 0:
            print 'AVM.fit %s %s %s' % (
                self.model_name, self.forecast_time_period, str(df_kept.shape))
        X_train, y_train = self.extract_and_transform(df_kept)
        return {
            'ElasticNet': fit_elastic_net,
            'RandomForestRegressor': fit_random_forest_regressor,
        }[self.model_name](X_train, y_train)

    def get_attributes(self):
        'return both sets of attributes, with None if not used by that model'
        pdb.set_trace()
        attribute_names = (
            'coef_', 'sparse_coef_', 'intercept_', 'n_iter_',                        # for linear
            'estimators_', 'feature_importances_', 'oob_score_', 'oob_prediction_',  # for random forest
        )
        return {name: getattr(self.model, name, None) for name in attribute_names}

    # Rely on the parent class to implement get_params and set_params
    # However, we may need a get_params method to support cloning; TODO: ask AM

    def extract_and_transform(self, df):
        'return X and y'
        def extract_and_transform_elastic_net(df):
            f = Features()
            return f.extract_and_transform_X_y(
                df,
                f.ege(),
                layout_transactions.price,
                self.units_X,
                self.units_y,
            )

        def extract_and_transform_random_forest_regressor(df):
            f = Features()
            return f.extract_and_transform_X_y(
                df,
                f.ege(),
                layout_transactions.price,
                'natural',
                'natural',
            )

        return {
            'ElasticNet': extract_and_transform_elastic_net,
            'RandomForestRegressor': extract_and_transform_random_forest_regressor,
        }[self.model_name](df)

    def predict(self, df):
        def predict_elastic_net(df):
            if self.verbose > 0:
                print 'predict_elastic_net'
            X_test, y_test = self.extract_and_transform(df)
            return self.model.predict(X_test)

        def predict_random_forest_regressor(df):
            if self.verbose > 0:
                print 'predict_random_forest_regressor'
            X_test, y_test = self.extract_and_transform(df)
            return self.model.predict(X_test)

        return {
            'ElasticNet': predict_elastic_net,
            'RandomForestRegressor': predict_random_forest_regressor,
        }[self.model_name](df)

    def setattr(self, parameter, value):
        setattr(self, parameter, value)
        return self


if False:
    pd()
    pprint()
    Features()
