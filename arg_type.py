'''type verifies for argparse

each function either
- returns an argument parsed from a string (possible the string); OR
- raises argpare.ArgumentTypeError
'''

import argparse
import pdb

if False:
    pdb


def features(s):
    's is a name for a group of features'
    allowed = ('s', 'sw', 'swp', 'swpn')
    try:
        assert s in allowed
        return s
    except:
        raise argparse.ArgumentTypeError('%s is not one of %s' % (s, allowed))


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
    allowed = ('all', 'best1')
    try:
        assert s in allowed, s
        return s
    except:
        raise argparse.ArgumentTypeError('%s is not one of %s' % (s, allowed))


def locality(s):
    allowed = ('census', 'city', 'global', 'zip')
    try:
        assert s in allowed
        return s
    except:
        raise argparse.ArgumentTYpeError('%s is not one of %s', (s, allowed))


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
        raise argparse.ArgumentTypeError('%s is not a yearmonth' % s)


def positive_int(s):
    'convert s to a positive integer'
    try:
        value = int(s)
        assert value > 0
        return value
    except:
        raise argparse.ArgumentTypeError('%s is not a positive integer' % s)
