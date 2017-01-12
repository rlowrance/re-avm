'''functions to manipulate dictionaries of hyperparameters'''
import pdb
import unittest


values = {
    'alpha': (0.01, 0.03, 0.1, 0.3, 1.0),
    'l1_ratio': (0.0, 0.25, 0.50, 0.75, 1.0),
    'learning_rate': (.10, .25, .50, .75, .99),
    'max_depth': (1, 3, 10, 30, 100, 300),
    'max_features': (1, 'log2', 'sqrt', 'auto'),
    'n_estimators': (10, 30, 100, 300),
    'n_months_back': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 24),
    'units_X': ('natural', 'log'),
    'units_y': ('natural', 'log'),
}


def to_str(d):
    'return string with hyperparameters in canonical order'
    pdb.set_trace()
    print d
    result = ''
    for name in values.keys():
        spacer = '-' if len(result) > 0 else ''
        if name in d:
            value = d[name]
            print name, value
            if isinstance(value, float):
                # assure no loss of precision
                printed_value = '%4.2f' % value
                assert float(printed_value) == value, (value, printed_value)
            elif isinstance(value, int):
                printed_value = '%d' % value
                assert int(printed_value) == value, (value, printed_value)
            else:
                printed_value = value
            result += spacer + printed_value
        else:
            result += spacer
        print result
    pdb.set_trace()
    return result


def to_dict(s):
    'return dictionary d, where s was created by to_str(d)'
    pdb.set_trace()
    hp_names = values.keys()
    values = s.split('-')
    result = {}
    for i, value in enumerate(values):
        if value != '':
            name = hp_names[i]
            stored_value = (
                value if name in ('units_X', 'units_y') else
                float(value) if name in ('alpha', 'l1_ratio') else
                int(value)
            )
            result[name] = stored_value
    pdb.set_trace()
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
    for n_months_back in values['n_months_back']:
        for alpha in values['alpha']:
            for l1_ratio in values['l1_ratio']:
                for units_X in values['units_X']:
                    for units_y in values['units_y']:
                        yield {
                            'n_months_back': n_months_back,
                            'alpha': alpha,
                            'l1_ratio': l1_ratio,
                            'units_X': units_X,
                            'units_y': units_y,
                        }


def iter_hps_gb():
    for n_months_back in values['n_months_back']:
        for max_depth in values['max_depth']:
            for max_features in values['max_features']:
                for n_estimators in values['n_estimators']:
                    for learning_rate in values['learning_rate']:
                        yield {
                            'n_months_back': n_months_back,
                            'max_depth': max_depth,
                            'max_features': max_features,
                            'n_estimators': n_estimators,
                            'learning_rate': learning_rate,
                        }


def iter_hps_rf():
    for n_months_back in values['n_months_back']:
        for max_depth in values['max_depth']:
            for max_features in values['max_features']:
                for n_estimators in values['n_estimators']:
                    yield {
                        'n_months_back': n_months_back,
                        'max_depth': max_depth,
                        'max_features': max_features,
                        'n_estimators': n_estimators,
                    }


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

if __name__ == '__main__':
    unittest.main()
    if False:
        pdb
