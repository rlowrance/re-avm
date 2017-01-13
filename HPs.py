'''functions to manipulate dictionaries of hyperparameters'''
import collections
import pdb
import unittest


Entries = collections.namedtuple('Entries', 'values kind')

all = {
    'alpha': Entries((0.01, 0.03, 0.1, 0.3, 1.0), float),
    'l1_ratio': Entries((0.0, 0.25, 0.50, 0.75, 1.0), float),
    'learning_rate': Entries((.10, .25, .50, .75, .99), float),
    'max_depth': Entries((1, 3, 10, 30, 100, 300), int),
    'max_features': Entries((1, 'log2', 'sqrt', 'auto'), 'max_features'),
    'n_estimators': Entries((10, 30, 100, 300), int),
    'n_months_back': Entries((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 24), int),
    'units_X': Entries(('natural', 'log'), str),
    'units_y': Entries(('natural', 'log'), str),
}

names = sorted(all.keys())


def values(name):
    return all[name].values


def kind(name):
    return all[name].kind


def to_str_hp(name, value):
    'return string'
    k = kind(name)
    if k == float:
        result = '%4.2f' % value
        assert float(result) == value  # assure no loss of precision
    elif k == int:
        result = '%02d' % value
        assert int(result) == value
    elif k == str:
        result = value
    elif k == 'max_features':
        if value == 1:
            result = '%02d' % value
            assert int(result) == value
        else:
            result = value
    return result


def from_str_hp(name, s):
    'return value'
    k = kind(name)
    if k == 'max_features':
        if s == '1':
            result = 1
        else:
            result = s
    else:
        result = k(s)
    return result


def to_str(d):
    'return string with hyperparameters in alphabetic order'
    result = ''
    for i, name in enumerate(names):
        spacer = '-' if i > 0 else ''
        if name in d:
            result += spacer + to_str_hp(name, d[name])
        else:
            result += spacer
    return result


def from_str(s):
    'return dictionary d, where s was created by to_str(d)'
    result = {}
    pieces = s.split('-')
    assert len(pieces) == len(names), 'too few hyperparameters in string: %s' % s
    for i, s_value in enumerate(pieces):
        if s_value != '':
            name = names[i]
            result[name] = from_str_hp(name, s_value)
    return result


def iter_hps_model(model):
    'main entry point: yield dict of hp_name: value'
    assert model in ('en', 'gb', 'rf'), model
    if model == 'en':
        return iter_hps_en()
    elif model == 'gb':
        return iter_hps_gb()
    else:
        return iter_hps_rf()


def iter_hps_en():
    for n_months_back in values('n_months_back'):
        for alpha in values('alpha'):
            for l1_ratio in values('l1_ratio'):
                for units_X in values('units_X'):
                    for units_y in values('units_y'):
                        yield {
                            'n_months_back': n_months_back,
                            'alpha': alpha,
                            'l1_ratio': l1_ratio,
                            'units_X': units_X,
                            'units_y': units_y,
                        }


def iter_hps_gb():
    for n_months_back in values('n_months_back'):
        for max_depth in values('max_depth'):
            for max_features in values('max_features'):
                for n_estimators in values('n_estimators'):
                    for learning_rate in values('learning_rate'):
                        yield {
                            'n_months_back': n_months_back,
                            'max_depth': max_depth,
                            'max_features': max_features,
                            'n_estimators': n_estimators,
                            'learning_rate': learning_rate,
                            'units_X': 'natural',
                            'units_y': 'natural',
                        }


def iter_hps_rf():
    for n_months_back in values('n_months_back'):
        for max_depth in values('max_depth'):
            for max_features in values('max_features'):
                for n_estimators in values('n_estimators'):
                    yield {
                        'n_months_back': n_months_back,
                        'max_depth': max_depth,
                        'max_features': max_features,
                        'n_estimators': n_estimators,
                        'units_X': 'natural',
                        'units_y': 'natural',                 }


class TestAll(unittest.TestCase):
    def test_iter_en(self):
        verbose = False
        count = 0
        for hps in iter_hps_model('en'):
            count += 1
            if verbose:
                print count, hps
        self.assertEqual(1400, count)

    def test_iter_gb(self):
        verbose = False
        count = 0
        for hps in iter_hps_model('gb'):
            count += 1
            if verbose:
                print count, hps
        self.assertEqual(6720, count)

    def test_iter_rf(self):
        verbose = False
        count = 0
        for hps in iter_hps_model('rf'):
            count += 1
            if verbose:
                print count, hps
        self.assertEqual(1344, count)

    def test_to_str_dict(self):
        for hps in iter_hps_gb():
            filename_base = to_str(hps)
            d = from_str(filename_base)
            self.assertItemsEqual(hps, d)

if __name__ == '__main__':
    unittest.main()
    if False:
        pdb
