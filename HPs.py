'''functions to manipulate dictionaries of hyperparameters'''
import pdb
import unittest


class HPs(object):
    names = (
        'alpha',
        'l1_ratio',
        'learning_rate',
        'max_depth',
        'max_features',
        'n_estimators',
        'n_months_back',
        'units_X',
        'units_y',
    )

    def __init__():
        assert False, 'do not instantiate me'

    @staticmethod
    def to_str(d):
        'return string with hyperparameters in canonical order'
        pdb.set_trace()
        print d
        result = ''
        for name in HPs.names:
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

    @staticmethod
    def to_dict(s):
        'return dictionary d, where s was created by to_str(d)'
        pdb.set_trace()
        values = s.split('-')
        result = {}
        for i, value in enumerate(values):
            if value != '':
                name = HPs.names[i]
                stored_value = (
                    value if name in ('units_X', 'units_y') else
                    float(value) if name in ('alpha', 'l1_ratio') else
                    int(value)
                )
                result[name] = stored_value
        pdb.set_trace()
        return result

    @staticmethod
    def iter_hps_model(model):
        'main entry point: yield dict of hp_name: value'
        assert model in ('en', 'gb', 'rf'), model
        if model == 'en':
            return HPs.iter_hps_en()
        elif model == 'gb':
            return HPs.iter_hps_gb()
        else:
            return HPs.iter_hps_rf()

    # methods for models
    @staticmethod
    def iter_hps_en():
        for n_months_back in HPs.iter_n_months_back():
            for alpha in HPs.iter_alpha():
                for l1_ratio in HPs.iter_l1_ratio():
                    for units_X in HPs.iter_units_X():
                        for units_y in HPs.iter_units_y():
                            yield {
                                'n_months_back': n_months_back,
                                'alpha': alpha,
                                'l1_ratio': l1_ratio,
                                'units_X': units_X,
                                'units_y': units_y,
                            }

    @staticmethod
    def iter_hps_gb():
        for n_months_back in HPs.iter_n_months_back():
            for tree_hps in HPs.iter_hps_tree():
                for learning_rate in HPs.iter_learning_rate():
                    tree_hps.update({
                        'n_months_back': n_months_back,
                        'learning_rate': learning_rate
                    })
                    yield tree_hps

    @staticmethod
    def iter_hps_rf():
        for n_months_back in HPs.iter_n_months_back():
            for tree_hps in HPs.iter_hps_tree():
                tree_hps.update({
                    'n_months_back': n_months_back,
                })
                yield tree_hps

    @staticmethod
    def iter_hps_tree():
        for n_estimators in HPs.iter_n_estimators():
            for max_features in HPs.iter_max_features():
                for max_depth in HPs.iter_max_depth():
                    yield {
                        'n_estimators': n_estimators,
                        'max_features': max_features,
                        'max_depth': max_depth,
                    }

    # methods for individual hyperparameters
    @staticmethod
    def iter_alpha():
        'alpha multiplies the  penalty term in elastic net models'
        for alpha in (0.01, 0.03, 0.1, 0.3, 1.0):
            yield alpha

    @staticmethod
    def iter_l1_ratio():
        'for elastic net: 0 ==> L2 penalty only, 1 ==> L1 penalty only'
        for l1_ratio in (0.0, 0.25, 0.50, 0.75, 1.0):
            yield l1_ratio

    @staticmethod
    def iter_learning_rate():
        'for gradient bosting, the amount by which the contribution of the next tree is shrunk'
        for learning_rate in (.10, .25, .50, .75, .99):
            yield learning_rate

    @staticmethod
    def iter_max_depth():
        'for tree-based methods, maximum depth of an individual tree'
        for max_depth in (1, 3, 10, 30, 100, 300):
            yield max_depth

    @staticmethod
    def iter_max_features():
        'for tree-based methods, maximum number of features to consider when splitting a node'
        for max_features in (1, 'log2', 'sqrt', 'auto'):
            yield max_features

    @staticmethod
    def iter_n_estimators():
        'for tree-based methods, number of trees in the ensemble'
        for n_estimators in (10, 30, 100, 300):
            yield n_estimators

    @staticmethod
    def iter_n_months_back():
        'number of months of training data to use'
        for n_months_back in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 24):
            yield n_months_back

    @staticmethod
    def iter_units_X():
        'for elastic net, units for the X values'
        for units_X in ('natural', 'log'):
            yield units_X

    @staticmethod
    def iter_units_y():
        'for elastic net, units for the y values'
        for units_y in ('natural', 'log'):
            yield units_y


class TestHPs(unittest.TestCase):
    def test_iter_en(self):
        verbose = False
        count = 0
        for hps in HPs.iter_hps_model('en'):
            count += 1
            if verbose:
                print count, hps
        self.assertEqual(1400, count)

    def test_iter_gb(self):
        verbose = False
        count = 0
        for hps in HPs.iter_hps_model('gb'):
            count += 1
            if verbose:
                print count, hps
        self.assertEqual(6720, count)

    def test_iter_rf(self):
        verbose = False
        count = 0
        for hps in HPs.iter_hps_model('rf'):
            count += 1
            if verbose:
                print count, hps
        self.assertEqual(1344, count)

if __name__ == '__main__':
    unittest.main()
    if False:
        pdb
