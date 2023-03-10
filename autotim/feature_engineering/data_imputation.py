import pandas as pd


def fix_missing_data_points(df: pd.DataFrame, column_id: str, column_sort: str,
                            column_kind: str = None):
    """
    Performs data imputation for individual NaN data points within measurements:
        missing values are filled by backward fill, followed by a forward fill.
        Variables within multivariate time series are considered independent.

    :param pandas.DataFrame df: Dataframe, where missing values have to be filled
    :param str column_id: Name for the column with time series identifiers
    :param str column_sort: Name for the column containing time / chronological data
    :param str column_kind: Name for the column with multivariate time series
        variable names, default = None
    """
    tmp_df = df.copy()
    tmp_df.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)

    group_by_col_names = [column_id]
    if column_kind is not None:
        group_by_col_names.append(column_kind)

    tmp_df = tmp_df.sort_values([column_sort], ascending=[True])
    return tmp_df.groupby(group_by_col_names).apply(lambda group: group.bfill().ffill())


def imputation_train_time(df: pd.DataFrame, column_id: str, column_sort: str,
                          column_kind: str = None):
    """Performs imputation for missing values."""
    df = fix_missing_data_points(df=df, column_id=column_id, column_sort=column_sort,
                                 column_kind=column_kind)
    df.fillna(0, inplace=True)
    return df


def imputation_test_time(df: pd.DataFrame, ts_settings, column_id: str, column_sort: str,
                         column_kind: str = None, column_value: str = None):
    """Performs imputation for missing values, adds missing dimensions if applicable."""
    df = fix_missing_data_points(df=df, column_id=column_id, column_sort=column_sort,
                                 column_kind=column_kind)
    if column_kind is not None:
        df = fix_missing_column_kind(ts_settings=ts_settings, timeseries=df,
                                     column_id=column_id, column_kind=column_kind,
                                     column_value=column_value)
    else:
        cols_add = set(list(ts_settings)).difference(set(df.columns))
        for col_name in cols_add:
            df[col_name] = 0

    return df


def get_extra_rows(col_ids: list, extra_col_kinds: set,
                   id_col_name: str, kind_col_name: str, value_col_name: str):
    id_kind_measurement = []
    for col_id in col_ids:
        for kind in extra_col_kinds:
            id_kind_measurement.append([col_id, kind, 0])
    return pd.DataFrame(id_kind_measurement, columns=[id_col_name, kind_col_name, value_col_name])


def fix_missing_column_kind(ts_settings, timeseries, column_id: str, column_kind: str,
                            column_value: str):
    """
    Check if any column kinds are missing and adds mock empty column kind to a dataframe.

    Reason: if a column kind is missing entirely then features concerning it will not be built
            when h2o fills these features with NaN, then predictions are different
            from when these are mocked in with 0 values.
    """
    col_difference = set(list(ts_settings)) - set(timeseries[column_id].
                                                      drop_duplicates().values.tolist())

    if len(col_difference) > 0:
        pd_enriched = get_extra_rows(col_ids=timeseries[column_id].drop_duplicates()
                                        .values.tolist(), extra_col_kinds=col_difference,
                                     id_col_name=column_id, kind_col_name=column_kind,
                                     value_col_name=column_value)
        timeseries = pd.concat([timeseries, pd_enriched])

    return  timeseries
