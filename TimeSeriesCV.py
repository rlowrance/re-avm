import pdb
import sklearn
import sklearn.grid_search

import ege


class TimeSeriesCV(object):
    def __init__(self, estimator, param_grid_model_search,
                 scoring, time_periods, in_time_periods, make_X_y,
                 test=False, verbose=0):
        self.estimator = estimator
        self.param_grid_model_search = param_grid_model_search
        self.scoring = scoring  # lambda estimator, X, y --> number
        self.time_periods = time_periods  # list of objects; ex: (200401, 200402, ... )
        self.in_time_periods = in_time_periods  # lambda df, time_periods --> df
        self.make_X_y = make_X_y  # lambda df, time_period -> (X_array, y_vector)
        # ? need the X and y units: TODO put them in a model (only way to get it to work)
        self.test = test
        self.verbose = verbose  # number; trace if > 0

    def fit(self, df):
        'mutate self by creating attribute grid_scores_, a dictionary'
        # TODO: Ask AM about API: probably should by X, y, but that makes the API more complex
        #       as need to transform the X, y
        #       - extract from X, y the samples in one or more time periods
        #       - add features dependent on the age of the property (which depends on the time period)
        #       - maybe translate certain features and the target from natural to log units;
        #       The last two transformations must be done during the runtime of GridSearchCV
        pdb.set_trace()
        # examine each time period but the last
        grid_search_cv_results = {}
        for t in xrange(len(self.time_periods)):
            # train on data from periods 0, 1, ..., t - 1
            # test on data from period t
            train_time_periods = self.time_periods[0:(t + 1)]
            test_time_period = self.time_periods[t + 1]
            if self.verbose > 0:
                print 'time period', t
                print 'train_time_periods', train_time_periods
                print 'test_time_periods', test_time_period

            # build the data sets for time period t
            df_train = self.in_time_periods(df, train_time_periods)
            df_test = self.in_time_periods(df, [test_time_period])

            estimator = sklearn.grid_search.GridSearchCV(
                estimator=self.estimator,
                param_grid=self.param_grid_model_search,
                scoring=self.scoring,
                n_jobs=1 if self.test else -1,  # use all cores if not testing
                cv=2 if self.test else 10,      # number of folds
                verbose=self.verbose,
            )
            pdb.set_trace()  # step into this call
            # NOTE: args to fit must be X and y; not something convenient
            # NOTE: X and y depend on the test_time_period and on the units in
            # self.param_grid_model_search
            # could use age when sold, which can be precomputed without knowing
            # do the units transformation in the AVM.fit method
            # Just pass the data frame
            # then do feature selection the AVM.fit
            # BETTER: just call GridSearchCV, not this class
            train_X, train_y = ege.Make_X_y_function(df_train, test_time_period)  # not possible
            estimator.fit(train_X, train_y)
            test_X, test_y = ege.Make_X_y_function(df_test, test_time_period)
            estimator.score(test_X, test_y)

            estimator.score((df_test, test_time_period))  # may require 2 args

            # train_X, train_y = self.make_X_y(df_train, test_time_period)  # for now, don't create age-related features
            # test_X, test_y = self.make_X_y(df_test, test_time_period)
            # estimator.fit(train_X, train_y)
            # estimator.score(test_X, test_y)

            # save all the attributes from GridSearchCV
            grid_search_cv_results[t] = {
                'grid_scores_': estimator.grid_scores_,
                'best_estimator_': estimator.best_estimator_,
                'best_score_': estimator.best_score_,
                'best_params_': estimator.best_params_,
                'scorer_': estimator.scorer_,
            }

        pdb.set_trace()
        # set attributes
        # for now, just return the attributes from GridSearchCV and scorer_
        # maybe later, also determine the best estimator, score, and params
        self.grid_scores_ = grid_search_cv_results
        self.scorer_ = self.scoring   # check with AM if this is what to return
        return self  # as required by API
