'''gradient boosting regressor module for AVM class'''

import pdb
import sklearn

from Features import Features
import layout_transactions


def fit(avm, X_train, y_train):
    if avm.verbose > 0:
        print (
            'fit gradient boosting regressor',
            avm.loss,
            avm.learning_rate,
            avm.n_estimators,
            avm.max_depth,
            avm.max_features,
            avm.random_state,
        )

    avm.model = sklearn.ensemble.GradientBoostingRegressor(
        loss=avm.loss,
        learning_rate=avm.learning_rate,
        n_estimators=avm.n_estimators,
        max_depth=avm.max_depth,
        max_features=avm.max_features,
        random_state=avm.random_state,
    )
    avm.model.fit(X_train, y_train)
    return avm


def extract_and_transform(avm, df, transform_y):
    f = Features()
    return f.extract_and_transform_X_y(
        df,
        f.ege(avm.features_group),
        layout_transactions.price,
        'natural',
        'natural',
        transform_y,
    )


def predict(avm, X_test):
    return avm.model.predict(X_test)


if __name__ == '__main__':
    pdb
