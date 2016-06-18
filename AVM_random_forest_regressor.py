'''random forests regressor modules for AVM class

These are separate in order to reflect dependencies in the Makefile
'''

import pdb
import sklearn

from Features import Features
import layout_transactions


def fit(avm, X_train, y_train):
    if avm.verbose > 0:
        print (
            'fit random forest regressor',
            avm.forecast_time_period,
            avm.n_estimators,
            avm.max_depth,
            avm.max_features,
            avm.n_months_back,
        )
    avm.model = sklearn.ensemble.RandomForestRegressor(
        n_estimators=avm.n_estimators,
        max_depth=avm.max_depth,
        random_state=avm.random_state,
        max_features=avm.max_features,
    )
    avm.model.fit(X_train, y_train)
    return avm


def extract_and_transform(avm, df, transform_y):
    f = Features()
    return f.extract_and_transform_X_y(
        df,
        f.ege(),
        layout_transactions.price,
        'natural',
        'natural',
        transform_y,
    )


def predict(avm, X_test):
    if avm.verbose > 0:
        print 'predict_random_forest_regressor'
    return avm.model.predict(X_test)


if __name__ == '__main__':
    if False:
        # avoid warning messages from checkers
        pdb()
