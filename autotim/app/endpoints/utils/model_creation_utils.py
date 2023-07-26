import os

import logging
import shutil

import numpy as np
import pandas as pd

from h2o.exceptions import H2OServerError

from autotim.feature_engineering.automated_feature_engineering import create_features, \
    select_relevant_features
from autotim.feature_engineering.exceptions import FeatureCreationFailedError

from autotim.model_training.autotim_training import AutoTiMTrainer

from autotim.model_selection.model_selection import ModelSelector
from autotim.model_selection.exceptions import ModelSelectionFailed, \
    MlflowExperimentNotFoundError


def merge_response_dict(versions, warning: str):
    response = {'training': "completed"}

    if isinstance(versions, dict):
        response = {**response, **versions}
    elif versions is not None:
        response['latest_model_version'] = versions

    if warning != '':
        response['warning'] = warning
    return response


def get_labels(dataframe: pd.DataFrame):
    """
    Extracts unique label for each timeseries in a given dataframe.
    :param dataframe: dataset with the uniquely identifiable timeseries
    :returns: series containing one label per timeseries
    """
    ids = dataframe[os.getenv("COLUMN_ID")].unique()
    labels = []
    for identifier in ids:
        label = dataframe.loc[dataframe[os.getenv("COLUMN_ID")] == identifier] \
            .iloc[0][os.getenv("COLUMN_LABEL")]
        labels.append(label)
    return pd.Series(data=labels, index=ids)


def train_test_split(dataset: pd.DataFrame, train_size:int):
    """Splits dataset into train and test subset"""

    ids = dataset[os.getenv("COLUMN_ID")].unique()
    np.random.shuffle(ids)

    train_ids = list(ids[:int(len(ids) * train_size)])

    train_df = dataset[dataset[os.getenv("COLUMN_ID")].isin(train_ids)]
    test_df = dataset[~dataset[os.getenv("COLUMN_ID")].isin(train_ids)]

    X_train = train_df.drop(columns=[os.getenv("COLUMN_LABEL")])
    y_train = get_labels(train_df)
    X_test = test_df.drop(columns=[os.getenv("COLUMN_LABEL")])
    y_test = get_labels(test_df)
    return X_train, y_train, X_test, y_test


def read_and_check_dataset(path: str):
    logging.debug(f"Reading {path} ...")

    dataset = pd.read_csv(path)

    for key in ['COLUMN_ID', 'COLUMN_LABEL', 'COLUMN_VALUE', 'COLUMN_KIND']:
        if os.getenv(key, "") != "" and not os.getenv(key) in dataset.columns:
            raise ValueError(f"Dataset does not contain the column '{os.getenv(key)}' that was set as {key}")
    return dataset


def extract_features(x_train, y_train):
    logging.debug("Extracting features for training ...")
    features_train = create_features(
        x_train, y_train, column_id=os.getenv(
            'COLUMN_ID'),
        column_value=os.getenv('COLUMN_VALUE') if os.getenv(
            'COLUMN_VALUE') != "" else None,
        column_kind=os.getenv('COLUMN_KIND') if os.getenv(
            'COLUMN_KIND') != "" else None
    )
    return features_train


def train_model(name, identifier, features_train, y_train):
    experiment_name = f"{name}-{identifier}"
    trainer = AutoTiMTrainer(experiment_name=experiment_name,
                            model_name=f"{experiment_name}_model")
    model_version = trainer.train(features_train, y_train)
    logging.info(f"Best {experiment_name} model has been logged as version: "
                 f"{str(model_version)}")

    return experiment_name, model_version


def select_model(name, identifier, model_version, experiment_name, X_test, y_test):
    old_metric, warning = None, ''
    new_metric = os.getenv('METRIC')

    model_selector = ModelSelector(name=name, identifier=identifier,
                                   latest_model_version=model_version)
    if model_selector.production_artifact_unavailable:
        warning = f"Checkpoints or features for the {experiment_name} model, " \
                  f"that was last set to Production were not found. " \
                  f"No model selection was conducted and the latest model was set to " \
                  f"Production."
    logging.debug("Updating production flag...")
    model_selector.update_production_flag(
        x_test=X_test, y_test=y_test, metric=new_metric)
    logging.debug("Checking if metric has changed...")
    old_metric = model_selector.metric_has_changed(
        metric=new_metric)
    return old_metric, new_metric, warning


def dataset_split(path, evaluation_identifier, dataset, train_size):
    if evaluation_identifier is not None:
        eval_data = read_and_check_dataset(path)

        logging.debug("Train/Test Split ...")
        # pylint: disable=no-member
        x_train = dataset.drop(columns=[os.getenv("COLUMN_LABEL")])
        y_train = get_labels(dataset)
        x_test = eval_data.drop(columns=[os.getenv("COLUMN_LABEL")])
        y_test = get_labels(eval_data)
        # pylint: enable=no-member
    else:
        logging.debug("Train/Test Split ...")
        x_train, y_train, x_test, y_test = train_test_split(dataset, train_size=train_size)

    return x_train, x_test, y_train, y_test


def train(name: str, identifier: str, path: str, train_size,
          max_attempts, evaluation_identifier=None):
    """Run training for the given use case."""
    dataset = read_and_check_dataset(path)

    x_train, x_test, y_train, y_test = \
        dataset_split(path=path,
                    evaluation_identifier=evaluation_identifier,
                    dataset=dataset, train_size=train_size)

    # Extract Features
    features_train = extract_features(x_train, y_train)

    warning = ''
    old_metric, new_metric = '', ''
    for attempt_model_creation in range(int(max_attempts)):

        features_train = select_relevant_features(features=features_train,
                                                  target_vector=y_train,
                                                  features_decrement_count=
                                                  attempt_model_creation)

        # Train Model
        experiment_name, model_version = train_model(
            name, identifier, features_train, y_train)

        # Predict
        try:
            old_metric, new_metric, warning = select_model(
                name, identifier, model_version, experiment_name, x_test, y_test)
            break
        except (ModelSelectionFailed, MlflowExperimentNotFoundError) as e:
            logging.error(e)
            warning = f"Logged experiment or model was not found when trying to select the best " \
                f"performing model for this use case and dataset combination. " \
                f"Please double-check manually if model {experiment_name}, " \
                f"version {model_version} has been persisted in MlFlow."
        except (H2OServerError, FeatureCreationFailedError) as e:
            logging.error(e)
            warning = "Model evaluation / best model selection failed due to an internal error. " \
                "We cannot guarantee that the best model is set to Production, " \
                "please check this manually in the MlFlow database."
        except RecursionError as e:
            logging.error(e)
            if attempt_model_creation == int(max_attempts) - 1:
                return {'training': "failed",
                                'error': "Internal error during training occurred." +
                                "Please define the paramater max_attempts " +
                                "from default '5' to an higer number " +
                                "in your request. For more information " +
                                "have a look into the README.md"}
            logging.warning(
                "Too many features for model prediction. Retrain model with less features")

    # TODO:
    # Clear data folder
    data_folder = "data_test"
    if os.path.exists(data_folder):
        shutil.rmtree(data_folder)

    if old_metric != "" and new_metric != "" and old_metric is not None and new_metric is not None \
            and old_metric != new_metric:
        warning = f"The chosen metric has changed from {str(old_metric)} to " \
                  f"{str(new_metric)}. Changing metrics between experiment runs " \
                  f"is not encouraged. To guarantee that the desired model is set to " \
                  f"production, please do so manually."

    return merge_response_dict(versions=model_version,
                                       warning=warning)
