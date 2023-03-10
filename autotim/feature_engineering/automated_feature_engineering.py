"""Automatically extract relevant features. """
import os

from tsfresh import extract_features, extract_relevant_features
from tsfresh.feature_selection.relevance import calculate_relevance_table
from tsfresh.utilities.dataframe_functions import impute

import h2o
from h2o.exceptions import H2OResponseError

from autotim.feature_engineering.exceptions import FeatureCreationFailedError
from autotim.feature_engineering.data_imputation import imputation_test_time, \
    imputation_train_time


def load_features_from_settings(data, column_id=None, column_value=None, column_kind=None,
                                settings=None):
    """Load features."""
    features = extract_features(data, column_id=column_id,
                                column_value=column_value,
                                column_kind=column_kind,
                                kind_to_fc_parameters=settings,
                                impute_function=impute)
    features = h2o.H2OFrame(features)
    #check if header of pd.DataFrame was converted to H2OFrame, then remove
    if len(features) == len(set(data[column_id].unique()))+1:
        features = features.drop([0], axis=0)

    return features

# Select the most important features with the lowest p_value
def select_relevant_features(features, target_vector, features_decrement_count):
    relevance_table = calculate_relevance_table(features, target_vector)
    sorted_relevance_table = relevance_table.sort_values(by=['p_value'])
    max_features = int(os.getenv("MAX_FEATURES"))
    features_decrement = float(os.getenv("FEATURES_DECREMENT"))

    limit = int(max_features * (features_decrement**features_decrement_count)) \
        if features_decrement < 1 \
        else int(max_features - (features_decrement**features_decrement_count))
    trimmed_relevance_table = sorted_relevance_table.iloc[:limit]

    names_of_selected_features = trimmed_relevance_table['feature'].to_list()
    selected_features = features[names_of_selected_features]
    return selected_features


def create_features(dataframe, var_y=None, column_id=None,
                    column_value=None, column_kind=None, column_sort="time", settings=None):
    try:
        if settings:
            dataframe = imputation_test_time(df=dataframe, ts_settings=settings,
                                             column_id=column_id, column_sort=column_sort,
                                             column_kind=column_kind, column_value=column_value)
            features = load_features_from_settings(data=dataframe,
                                                   column_id=column_id,
                                                   column_value=column_value,
                                                   column_kind=column_kind,
                                                   settings=settings)
        else:
            dataframe = imputation_train_time(df=dataframe, column_id=column_id,
                                              column_sort=column_sort, column_kind=column_kind)

            features = extract_relevant_features(dataframe, var_y,
                                                 column_id=column_id,
                                                 column_sort=column_sort,
                                                 column_value=column_value,
                                                 column_kind=column_kind)

            # if no relevant features were extracted, then perform feature calculation
                                                                    # without the relevance test
            if features.shape[1] == 0:
                features = extract_features(dataframe, column_id=column_id,
                    column_sort=column_sort, column_value=column_value, column_kind=column_kind)

            features = impute(features)
        return features
    except (ValueError, H2OResponseError, AssertionError) as e:
        # ValueError -> empty dataframe
        # H2OResponseError -> dataframe not empty, but wrong format
        # AssertionError -> settings are None or an empty dictionary
        raise FeatureCreationFailedError(message=str(e)) from e
