import numpy as np
import pandas as pd
import pdb
from pprint import pprint
import unittest

from Month import Month


def make_test_train(test_time_period, train_n_months_back, trade_month_column_name, samples):
    'return dataframes for testing and training; see valavm.do_val.fit_and_run'
    assert isinstance(test_time_period, Month), test_time_period
    trade_month = samples[trade_month_column_name]
    assert trade_month.dtype == np.dtype('int64')
    test_month = test_time_period.as_int()

    test_mask = trade_month == test_month
    test_df = samples[test_mask]

    first_train_month = Month(test_time_period).decrement(train_n_months_back)
    print 'first_train_month', first_train_month
    train_mask = (first_train_month.as_int() <= trade_month) & (trade_month < test_month)
    train_df = samples[train_mask]

    assert len(test_df) > 0
    assert len(train_df) > 0

    return test_df, train_df


class Test_make_test_train(unittest.TestCase):
    def test_1(self):
        def vp(x):
            if False:
                pprint(x)
        yyyymm = 'trade_month'
        x = 'x'
        samples = pd.DataFrame([
            {yyyymm: 200702, x: 0},
            {yyyymm: 200701, x: 1},
            {yyyymm: 200612, x: 2},
            {yyyymm: 200611, x: 3},
            {yyyymm: 200610, x: 4},
        ])
        vp(samples)
        test, train = make_test_train(Month(200702), 1, yyyymm, samples)
        vp(test)
        vp(train)
        assert len(test) == 1
        assert len(train) == 1
        test, train = make_test_train(Month(200701), 2, yyyymm, samples)
        vp(test)
        vp(train)
        assert len(test) == 1
        assert len(train) == 2
        pass


if __name__ == '__main__':
    unittest.main()
    if False:
        pdb.set_trace()
