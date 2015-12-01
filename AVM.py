'Automated Valuation Model'

import pdb
import numpy as np
import pandas as pd
from pprint import pprint
import sklearn
import sklearn.ensemble
import sklearn.linear_model
import sklearn.preprocessing


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
                 loss=None,                # for GradientBoostingRegressor
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

            assert self.alpha > 0.0, self.alpha  # otherwise, not reliable
            self.model = sklearn.linear_model.ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                fit_intercept=True,
                normalize=True,
                selection='random',   # select random coefficient to update at each iteration
                random_state=self.random_state,
            )
            self.scaler = sklearn.preprocessing.MinMaxScaler()
            self.scaler.fit(X_train)
            X_scaled = self.scaler.transform(X_train)
            self.model.fit(X_scaled, y_train)
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
        if False and ((self.units_X == 'log') or (self.units_y == 'log')):
            print self.units_X, self.units_y
            print 'check transformation to log'
            pdb.set_trace()
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

    def extract_and_transform(self, df, transform_y=True):
        'return X and y'
        def extract_and_transform_elastic_net(df):
            f = Features()
            return f.extract_and_transform_X_y(
                df,
                f.ege(),
                layout_transactions.price,
                self.units_X,
                self.units_y,
                transform_y,
            )

        def extract_and_transform_random_forest_regressor(df):
            f = Features()
            return f.extract_and_transform_X_y(
                df,
                f.ege(),
                layout_transactions.price,
                'natural',
                'natural',
                transform_y,
            )

        return {
            'ElasticNet': extract_and_transform_elastic_net,
            'RandomForestRegressor': extract_and_transform_random_forest_regressor,
        }[self.model_name](df)

    def predict(self, df):
        def predict_elastic_net(X_test):
            if self.verbose > 0:
                print 'predict_elastic_net'
            X_scaled = self.scaler.transform(X_test)
            answer_raw = self.model.predict(X_scaled)
            answer = answer_raw if self.units_y == 'natural' else np.exp(answer_raw)
            if False:
                pdb.set_trace()
                print 'elastic net', self.units_X, self.units_y
                print answer[:10]
            return answer

        def predict_random_forest_regressor(X_test):
            if self.verbose > 0:
                print 'predict_random_forest_regressor'
            return self.model.predict(X_test)

        X_test, y_test = self.extract_and_transform(df, transform_y=False)
        assert y_test is None
        return {
            'ElasticNet': predict_elastic_net,
            'RandomForestRegressor': predict_random_forest_regressor,
        }[self.model_name](X_test)

    def setattr(self, parameter, value):
        setattr(self, parameter, value)
        return self


if False:
    pd()
    pprint()
    Features()
