'''type verifiers for argparse

each function either
- returns an argument parsed from a string (possible the string); OR
- raises argpare.ArgumentTypeError
'''

import argparse
import multiprocessing
import os
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
    return s in set('all', 'best1')


def _in_set(s, allowed):
    'return s or raise ArgumentTypeError'
    try:
        assert s in allowed
        return s
    except:
        raise argparse.ArgumentTypeError('s not in allowed values {%s}' (s, allowed))

locality_choices = set(['census', 'city', 'global', 'zip'])

model_choices = set(['en', 'gb', 'rf'])


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
    if s == 'global':
        return s
    else:
        return s.replace('_', ' ')


def n_cities(s):
    return positive_int(s)


def n_processes(s):
    'return int value of s, if it is valid for system we are running on'
    cpu_count = multiprocessing.cpu_count()
    try:
        result = int(s)
        assert 1 <= result <= cpu_count
        return result
    except:
        raise argparse.ArgumentTypeError('%s not an itteger in [1,%d]' % (s, cpu_count))


def path_creatable(s):
    'is is a path to a file that can be created'
    try:
        # I can't get the statement below to work
        # assert os.access(s, os.W_OK)
        return s
    except:
        raise argparse.ArgumentTypeError('%s is not a path to a creatable file' % s)


def path_existing(s):
    's is a path to an existing file or directory'
    try:
        assert os.path.exists(s)
        return s
    except:
        raise argparse.ArgumentTypeError('%s is not a path to an existing file or dir' % s)


def positive_int(s):
    'convert s to a positive integer or raise exception'
    try:
        value = int(s)
        assert value > 0
        return value
    except:
        raise argparse.ArgumentTypeError('%s is not a positive integer' % s)


training_data_choices = set(['all', 'train'])


def year(s):
    'convert s to integer that could be a year'
    try:
        assert len(s) == 4
        value = int(s)
        return value
    except:
        raise argparse.ArgumentTypeError('%s is not a year' % s)