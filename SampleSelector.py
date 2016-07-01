'select portions of samples'

import numpy as np
import pdb

import layout_transactions


class SampleSelector(object):
    def __init__(self, samples):
        self.samples = samples.copy()
        self.dates = samples[layout_transactions.yyyymm]
        assert isinstance(self.dates[0], np.int64), self.dates

    def in_month(self, month):
        mask = self.dates == month.as_int()
        kept = self.samples[mask]
        return kept

    def between_months(self, first, last):
        mask_first = self.dates >= first.as_int()
        mask_last = self.dates <= last.as_int()
        mask = mask_first & mask_last
        kept = self.samples[mask]
        return kept


if __name__ == '__main__':
    # TODO: write unit test
    if False:
        pdb
