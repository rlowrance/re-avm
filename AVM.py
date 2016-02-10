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
import AVM_elastic_net
import AVM_gradient_boosting_regressor
import AVM_random_forest_regressor
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
    'one estimator for several underlying models'
    def __init__(self,
                 model_name=None,          # parameters for all models
                 forecast_time_period=None,
                 n_months_back=None,
                 random_state=None,
                 verbose=0,
                 implementation_module=None,
                 alpha=None,               # for ElasticNet
                 l1_ratio=None,
                 units_X=None,
                 units_y=None,
                 n_estimators=None,        # for RandomForestRegressor
                 max_depth=None,
                 max_features=None,
                 learning_rate=None,       # for GradientBoostingRegressor
                 loss=None,
                 ):
        # NOTE: just capture the parameters (to conform to the sklearn protocol)
        self.model_name = model_name
        self.forecast_time_period = forecast_time_period
        self.n_months_back = n_months_back
        self.random_state = random_state
        self.verbose = verbose
        self.implementation_module = implementation_module

        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.units_X = units_X
        self.units_y = units_y

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features

        self.learning_rate = learning_rate
        self.loss = loss

    def fit(self, samples):
        'convert samples to X,Y and fit them'
        self.implementation_module = {
            'ElasticNet': AVM_elastic_net,
            'GradientBoostingRegressor': AVM_gradient_boosting_regressor,
            'RandomForestRegressor': AVM_random_forest_regressor,
        }[self.model_name]
        X_train, y_train = self.extract_and_transform(samples)
        self.implementation_module.fit(self, X_train, y_train)

    def fitOLD(self, df):
        'construct and fit df that contains X and y'
        assert False, 'deprecated'
        self.implementation_module = {
            'ElasticNet': AVM_elastic_net,
            'GradientBoostingRegressor': AVM_gradient_boosting_regressor,
            'RandomForestRegressor': AVM_random_forest_regressor,
        }[self.model_name]
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
        self.implementation_module.fit(self, X_train, y_train)

    def get_attributes(self):
        'return both sets of attributes, with None if not used by that model'
        pdb.set_trace()
        attribute_names = (
            'coef_', 'sparse_coef_', 'intercept_', 'n_iter_',                        # for linear
            'estimators_', 'feature_importances_', 'oob_score_', 'oob_prediction_',  # for random forest
        )
        return {name: getattr(self.model, name, None) for name in attribute_names}

    def extract_and_transform(self, samples, transform_y=True):
        'return X and y'
        return self.implementation_module.extract_and_transform(self, samples, transform_y)

    def predict(self, samples):
        X_test, y_test = self.extract_and_transform(samples, transform_y=False)
        assert y_test is None
        return self.implementation_module.predict(self, X_test)

    def setattr(self, parameter, value):
        setattr(self, parameter, value)
        return self


if False:
    pd()
    pprint()
    Features()
