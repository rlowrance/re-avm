import pdb
import unittest


class Fitted(object):
    def __init__(self, training_data, neighborhood, model):
        assert training_data in Fitted.training_data_choices()
        assert neighborhood == 'global' or neighborhood in Fitted.selected_cities_choices()
        assert model in Fitted.model_choices()
        self.training_data = training_data
        self.neighborhood = neighborhood
        self.model = model

    def __repr__(self):
        return "Fitted('%s','%s','%s')" % (self.training_data, self.neighborhood, self.model)

    @staticmethod
    def model_choices():
        return ('en', 'gb', 'rf')

    @staticmethod
    def selected_cities_choices():
        'return city names selected by selected_cities'
        return (
            'WESTLAKE VILLAGE',  # from selected-cities2
            'HARBOR CITY',
            'ARTESIA',
            'MALIBU',
            'WILMINGTON',
            'VENICE',
            'PARAMOUNT',
            'AGOURA HILLS',
            'PASADENA',
            'WHITTIER',
            'LONG BEACH',
            'LOS ANGELES',
        )

    @staticmethod
    def training_data_choices():
        return ('train', 'all')


class TestFitted(unittest.TestCase):
    def test_construction_ok(self):
        for test in (
            ('train', 'global', 'en'),
            ('all', 'VENICE', 'rf'),
            ('train', 'LOS ANGELES', 'gb'),
        ):
            training_data, neighborhood, model = test
            x = Fitted(training_data, neighborhood, model)
            self.assertTrue(isinstance(x, Fitted))
            self.assertEqual(training_data, x.training_data)
            self.assertEqual(neighborhood, x.neighborhood)
            self.assertEqual(model, x.model)

    def test_construction_bad(self):
        for test in (
            ('something', 'global', 'en'),
            ('all', 'WEST HOLLYWOOD', 'gb'),
            ('all', 'global', 'xx'),
        ):
            training_data, neighborhood, model = test
            self.assertRaises(AssertionError, Fitted, training_data, neighborhood, model)

    def test_repr(self):
        verbose = False
        for test in (
            ('train', 'global', 'en'),
        ):
            a, b, c = test
            x = Fitted(a, b, c)
            s = '%s' % x
            if verbose:
                print s


if __name__ == '__main__':
    unittest.main()
    if False:
        pdb  # avoid warning form pyflake8
