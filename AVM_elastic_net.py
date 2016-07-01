'''elastic net modules for AVM class

These are separate in order to reflect dependencies in the Makefile
'''

import pdb
import numpy as np
import sklearn

from Features import Features
import layout_transactions


def fit(avm, X_train, y_train):
    if avm.verbose > 0:
        print 'fit elastic net: %s~%s alpha: %f l1_ratio: %f' % (
            avm.units_X, avm.units_y, avm.alpha, avm.l1_ratio)

    assert avm.alpha > 0.0, avm.alpha  # otherwise, not reliable
    avm.model = sklearn.linear_model.ElasticNet(
        alpha=avm.alpha,
        l1_ratio=avm.l1_ratio,
        fit_intercept=True,
        normalize=True,
        selection='random',   # select random coefficient to update at each iteration
        random_state=avm.random_state,
    )
    avm.scaler = sklearn.preprocessing.MinMaxScaler()
    avm.scaler.fit(X_train)
    X_scaled = avm.scaler.transform(X_train)
    avm.model.fit(X_scaled, y_train)
    return avm


def extract_and_transform(avm, df, transform_y):
    f = Features()
    return f.extract_and_transform_X_y(
        df,
        f.ege(avm.features_group),
        layout_transactions.price,
        avm.units_X,
        avm.units_y,
        transform_y,
    )


def predict(avm, X_test):
    if avm.verbose > 0:
        print 'predict_elastic_net'
    X_scaled = avm.scaler.transform(X_test)
    answer_raw = avm.model.predict(X_scaled)
    answer = answer_raw if avm.units_y == 'natural' else np.exp(answer_raw)
    if False:
        pdb.set_trace()
        print 'elastic net', avm.units_X, avm.units_y
        print answer[:10]
    return answer
