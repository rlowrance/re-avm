'Automated Valuation Model'

import pdb
from pprint import pprint
import sklearn


class AVM(sklearn.base.BaseEstimator):
    'one estimator for two underlying models: ElasticNet and RandomForestRegressor'
    def __init__(self,
                 alpha=1.0,
                 l1_ratio=0.5,                # for ElasticNet
                 n_estimators=10,             # for RandomForestRegressor
                 max_depth=10,                # for RandomForestRegressor
                 model_family='elastic net',   # for all
                 column_dict={},  # key = feature name, value = column index in X
                 last_training_yyyymm=200902,
                 n_months_back=1,
                 random_state=None,
                 units_X='natural',
                 units_y='natural',
                 time_periods_test=(200902,),
                 time_periods_train=(200901,),
                 transformer=None,
                 ):
        # NOTE: just capture the parameters (to conform to the sklearn protocol)
        # hyperparameters for linear models
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        # hyperparameters for tree models
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        # hyperparameters for all models
        self.model_family = model_family  # 'linear' or 'tree'
        self.column_dict = column_dict
        self.n_months_back = n_months_back  # number of months back to look with fitting data
        self.random_state = random_state
        self.last_training_yyyymm = last_training_yyyymm
        self.units_X = units_X
        self.units_y = units_y
        self.time_periods_train = time_periods_train
        self.time_periods_test = time_periods_test
        self.transformer = transformer

    def _fit_elastic_net(self, X_train, y_train):
        assert self.l1_ratio > 0.01, self.l1_ratio  # otherwise, not reliable
        self.model = sklearn.linear_model.ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            fit_intercept=True,
            normalize=True,
            random_state=self.random_state,
        )
        self.model.fit(X_train, y_train)
        return self

    def setattr(self, parameter, value):
        setattr(self, parameter, value)
        return self

    def _fit_random_forest(self, X_train, y_train):
        self.model = sklearn.ensemble.RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        self.model.fit(X_train, y_train)
        return self

    def fit(self, arg):
        'construct and fit df that contains X and y'
        pprint(arg, self)
        pdb.set_trace()
        df, time_period = arg
        X_train, y_train = self.make_X_y(df, time_period)
        if self.model_family == 'elastic net':
            return self._fit_elastic_net(X_train, y_train)
        elif self.model_family == 'random forest':
            return self._fit_random_forest(X_train, y_train)
        else:
            print 'bad model_family:', self.model_family

    def get_attributes(self):
        'return both sets of attributes, with None if not used by that model'
        pdb.set_trace()
        attribute_names = (
            'coef_', 'sparse_coef_', 'intercept_', 'n_iter_',                        # for linear
            'estimators_', 'feature_importances_', 'oob_score_', 'oob_prediction_',  # for random forest
        )
        return {name: getattr(self.model, name, None) for name in attribute_names}

    def predict(self, X):
        return self.model.predict(X)

    def set_params(self, **parameters):
        pdb.set_trace()
        for parameter, value in parameters.iteritems():
            self.setattr(parameter, value)
        return self
