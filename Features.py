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
        return (  # 28 features (added has_commercial)
            (t.age, None),
            (t.age2, None),
            (t.age_effective, None),
            (t.age_effective2, None),
            (t.building_basement_square_feet, 'log1p'),
            (t.building_baths, 'log1p'),
            (t.building_bedrooms, 'log1p'),
            (t.building_fireplace_number, 'log1p'),
            (t.building_has_pool, None),
            (t.building_is_new_construction, None),
            (t.building_living_square_feet, 'log'),
            (t.building_rooms, 'log1p'),
            (t.building_stories, 'log1p'),
            (t.census2000_fraction_owner_occupied, None),
            (t.census2000_median_household_income, 'log'),
            (t.census2000_avg_commute, None),
            (t.census_tract_has_commercial, None),
            (t.census_tract_has_industry, None),
            (t.census_tract_has_park, None),
            (t.census_tract_has_retail, None),
            (t.census_tract_has_school, None),
            (t.lot_land_square_feet, 'log'),
            (t.lot_parking_spaces, 'log1p'),
            (t.zip5_has_commercial, None),
            (t.zip5_has_industry, None),
            (t.zip5_has_retail, None),
            (t.zip5_has_park, None),
            (t.zip5_has_school, None),
        )
        return (  # 26 features in OLD version
            ('fraction.owner.occupied', None),
            ('FIREPLACE.NUMBER', 'log1p'),
            ('BEDROOMS', 'log1p'),
            ('BASEMENT.SQUARE.FEET', 'log1p'),
            ('LAND.SQUARE.FOOTAGE', 'log'),
            ('zip5.has.industry', None),
            ('census.tract.has.industry', None),
            ('census.tract.has.park', None),
            ('STORIES.NUMBER', 'log1p'),
            ('census.tract.has.school', None),
            ('TOTAL.BATHS.CALCULATED', 'log1p'),
            ('median.household.income', 'log'),  # not log feature in earlier version
            ('LIVING.SQUARE.FEET', 'log'),
            ('has.pool', None),
            ('zip5.has.retail', None),
            ('census.tract.has.retail', None),
            ('is.new.construction', None),
            ('avg.commute', None),
            ('zip5.has.park', None),
            ('PARKING.SPACES', 'log1p'),
            ('zip5.has.school', None),
            ('TOTAL.ROOMS', 'log1p'),
            ('age', None),
            ('age2', None),
            ('effective.age', None),
            ('effective.age2', None),
        )
