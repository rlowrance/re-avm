import math
import numpy as np
from pprint import pprint
import pdb

import layout_parcels as parcels
import layout_transactions as t


class Features(object):
    'return various feature sets'
    def __init__(self):
        '''in future, possibly read a file to determine some feature sets

        That possibility is why this functionality is in a class
        '''
        pass

    def ege(self):
        '''features needed for the ege_week program

        RETURN: tuple of (feature_name, how_to_transform_to_log_domain)

        Used to build the X array used for training
        The X array columns are in the order.
        '''
        non_geo_features = [
            (t.age, None),
            (t.age2, None),
            (t.age_effective, None),
            (t.age_effective2, None),
            (t.building_basement_square_feet, 'log1p'),
            (t.building_baths, 'log1p'),
            (t.building_bedrooms, 'log1p'),
            (t.building_fireplace_number, 'log1p'),
            (t.building_is_new_construction, None),
            (t.building_living_square_feet, 'log'),
            (t.building_rooms, 'log1p'),
            (t.building_stories, 'log1p'),
            (t.census2000_fraction_owner_occupied, None),
            (t.census2000_median_household_income, 'log'),
            (t.census2000_avg_commute, None),
            (t.has_pool, None),
            (t.lot_land_square_feet, 'log'),
            (t.lot_parking_spaces, 'log1p'),
        ]
        # these are all indicator variables (0 or 1)
        census_tract_propn = [('census_tract_has_' + x, None) for x in parcels.propn.keys()]
        zip5_propn = [('zip5_has_' + x, None) for x in parcels.propn.keys()]
        aggregated_features = ('any_commercial', 'any_industrial', 'any_non_residential')
        census_tract_aggregated = [('census_tract_has_' + x, None) for x in aggregated_features]
        zip5_aggregated = [('zip5_has_' + x, None) for x in aggregated_features]
        result = tuple(non_geo_features +
                       census_tract_propn +
                       zip5_propn +
                       census_tract_aggregated +
                       zip5_aggregated)
        return result

    def ege_names(self):
        'return names of features in order (not column names)'
        # NOTE: code is duplicated with the ege() method
        # So be careful if you modify either one
        non_geo_features = [
            'age', 'age2', 'age_effective', 'age_effective2',
            'building_basement_square_feet', 'building_baths', 'building_bedrooms',
            'building_fireplace_number', 'building_is_new_construction',
            'building_living_square_feet', 'building_rooms', 'building_stories',
            'census2000_fraction_owner_occupied', 'census2000_median_household_income',
            'census2000_avg_commute',
            'has_pool', 'lot_square_feet', 'lot_parking_spaces',
            ]
        # these are all indicator variables (0 or 1)
        census_tract_propn = ['census_tract_has_' + x for x in parcels.propn.keys()]
        zip5_propn = ['zip5_has_' + x for x in parcels.propn.keys()]
        aggregated_features = ('any_commercial', 'any_industrial', 'any_non_residential')
        census_tract_aggregated = ['census_tract_has_' + x for x in aggregated_features]
        zip5_aggregated = ['zip5_has_' + x for x in aggregated_features]
        result = tuple(
                non_geo_features +
                census_tract_propn +
                zip5_propn +
                census_tract_aggregated +
                zip5_aggregated)
        return result

    def extract_and_transform_X_y(self,
                                  df, features_transforms, target_feature_name,
                                  units_X, units_y,
                                  transform_y):
        'return X and y'
        def transform_series(value, how_to_transform=None):
            if how_to_transform is None:
                return value
            elif how_to_transform == 'log':
                return math.log(value)
            elif how_to_transform == 'log1p':
                return math.log1p(value)
            else:
                print 'bad how_to_transform:', how_to_transform
                pdb.set_trace()

        def transform_column(feature_name, how_to_transform, units):
            series = df[feature_name]
            transformed = (
                series if units == 'natural' else
                series.apply(transform_series, how_to_transform=how_to_transform)
            )
            return transformed.values

        X_transposed = np.empty((len(features_transforms), len(df),),
                                dtype='float64',
                                )
        for i, feature_transform in enumerate(features_transforms):
            feature_name, how_to_transform = feature_transform
            X_transposed[i] = transform_column(feature_name, how_to_transform, units_X)

        y = np.empty(len(df),
                     dtype='float64',
                     )
        y = (transform_column(target_feature_name, 'log', units_y)
             if transform_y
             else None)

        return X_transposed.T, y


if __name__ == '__main__':
    ege = Features().ege()
    print 'len(ege features)', len(ege)
    pprint(sorted(ege))
