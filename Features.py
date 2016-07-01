import math
import numpy as np
from pprint import pprint as pp
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

    def ege(self, features_group):
        '''return ((feature_name, transformation_function))'
        feature_names are the column names in the data frame
        transformation_functions are used to convert the feature value into the log domain
        '''
        assert features_group in ('s', 'sw', 'swp', 'swpn'), features_group
        s_features = (  # size features
            (t.building_living_square_feet, 'log'),
            (t.lot_land_square_feet, 'log'),
            )
        if features_group == 's':
            return s_features

        w_features = (  # weath of census tract features
            (t.census2000_fraction_owner_occupied, None),
            (t.census2000_median_household_income, 'log'),
            (t.census2000_avg_commute, None),
            )
        if features_group == 'sw':
            return s_features + w_features

        p_features = (  # property features
            (t.age, None),
            (t.age2, None),
            (t.age_effective, None),
            (t.age_effective2, None),
            (t.building_basement_square_feet, 'log1p'),
            (t.building_baths, 'log1p'),
            (t.building_bedrooms, 'log1p'),
            (t.building_fireplace_number, 'log1p'),
            (t.building_is_new_construction, None),
            (t.building_rooms, 'log1p'),
            (t.building_stories, 'log1p'),
            (t.has_pool, None),
            (t.lot_parking_spaces, 'log1p'),
            )
        if features_group == 'swp':
            return s_features + w_features + p_features

        census_tract_propn = [('census_tract_has_' + x, None) for x in parcels.propn.keys()]
        zip5_propn = [('zip5_has_' + x, None) for x in parcels.propn.keys()]
        aggregated_features = ['any_commercial', 'any_industrial', 'any_non_residential']
        census_tract_aggregated = [('census_tract_has_' + x, None) for x in aggregated_features]
        zip5_aggregated = [('zip5_has_' + x, None) for x in aggregated_features]
        if features_group == 'swpn':
            n_features = tuple(
                census_tract_propn +
                zip5_propn +
                census_tract_aggregated +
                zip5_aggregated
                )
            return s_features + w_features + p_features + n_features

    def ege_names(self, features_group):
        'return names of features in order (not column names)'
        feature_transformation = self.ege(features_group)
        result = [x[0] for x in feature_transformation]  # just the feature names
        return tuple(result)

    def extract_and_transform_X_y(self,
                                  df,
                                  features_transforms,
                                  target_feature_name,
                                  units_X,
                                  units_y,
                                  transform_y,
                                  ):
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
    # unit test
    for features_group in ('s', 'sw', 'swp', 'swpn'):
        feature_transform = Features().ege(features_group)
        print features_group, 'feature_transform'
        pp(feature_transform)
        feature = Features().ege_names(features_group)
        print features_group, 'feature names'
        pp(feature)
