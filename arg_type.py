'''type verifiers for argparse

each function either
- returns an argument parsed from a string (possible the string); OR
- raises argpare.ArgumentTypeError
'''

import argparse
import multiprocessing
import pdb

if False:
    pdb


def features(s):
    's is a name for a group of features'
    return _in_set(s, set('s', 'sw', 'swp', 'swpn'))


def features_hps(s):
    'return s or raise error'
    try:
        pieces = s.split('-')
        assert len(pieces) == 2, pieces
        # verify type of each piece
        maybe_features, maybe_hps = pieces
        features(maybe_features)
        hps(maybe_hps)
        return s
    except:
        raise argparse.ArgumentTypeError('%s is not a featuregroup-hps-yearmonth' % s)


def features_hps_month(s):
    'return s or raise error'
    try:
        pieces = s.split('-')
        assert len(pieces) == 3, pieces
        # verify type of each piece
        maybe_features, maybe_hps, maybe_month = pieces
        features(maybe_features)
        hps(maybe_hps)
        month(maybe_month)
        return s
    except:
        raise argparse.ArgumentTypeError('%s is not a featuregroup-hps-yearmonth' % s)


def features_hps_locality(s):
    'return s or raise error'
    try:
        pieces = s.split('-')
        assert len(pieces) == 3, pieces
        # verify type of each piece
        maybe_features, maybe_hps, maybe_locality = pieces
        features(maybe_features)
        hps(maybe_hps)
        locality(maybe_locality)
        assert pieces[2] in ['global', 'census', 'city', 'zip']
        return s
    except:
        raise argparse.ArgumentTypeError('%s is not a featuregroup-hps-locality' % s)


def features_hps_locality_month(s):
    'return s or raise error'
    try:
        pieces = s.split('-')
        pdb.set_trace()
        assert len(pieces) == 4, pieces
        # verify type of each piece
        features(pieces[0])
        hps(pieces[1])
        assert pieces[2] in ['global', 'census', 'city', 'zip']
        month(pieces[3])
        return s
    except:
        raise argparse.ArgumentTypeError('%s is not a featuregroup-hps-yearmonth-locality' % s)


def hps(s):
    's is the name of a group of hyperparameters'
    return in_set(s, set('all', 'best1'))


def _in_set(s, allowed):
    'return s or raise ArgumentTypeError'
    try:
        assert s in allowed
        return s
    except:
        raise argparse.ArgumentTypeError('s not in allowed values {%s}' (s, allowed))


def locality_choices(s):
    return set(['census', 'city', 'global', 'zip'])


def model_choices(s):
    return set(['en', 'gb', 'rf'])


def month(s):
    's is a string of the form YYYYMM'
    try:
        s_year = s[:4]
        s_month = s[4:]
        int_year = int(s_year)
        assert 0 <= int_year <= 2016
        int_month = int(s_month)
        assert 1 <= int_month <= 12
        return s
    except:
        raise argparse.ArgumentTypeError('%s is not a yearmonth of form YYYYMM' % s)


def neighborhood(s):
    's is global or a city name'
    # if a city name, replace _ by ' '
    return (
        s if s == 'global' else
        s.replace('_', ' ')
    )


def n_processes(s):
    cpu_count = multiprocessing.cpu_count()
    try:
        result = int(s)
        assert 1 <= result <= cpu_count
        return result
    except:
        raise argparse.ArgumentTypeError('%s not an itteger in [1,%d]' % (s, cpu_count))


def positive_int(s):
    'convert s to a positive integer or raise exception'
    try:
        value = int(s)
        assert value > 0
        return value
    except:
        raise argparse.ArgumentTypeError('%s is not a positive integer' % s)


def training_data_choices():
    return set(['all', 'train'])