import os

import logging
from pathlib import Path
import glob
import shutil

import numpy as np
import pandas as pd

from flask import jsonify
from flask_api import status

from h2o.exceptions import H2OError, H2OServerError
from mlflow.exceptions import MlflowException

from autotim.feature_engineering.automated_feature_engineering import create_features, \
    select_relevant_features
from autotim.feature_engineering.exceptions import FeatureCreationFailedError , \
    DataSplitError

from autotim.model_training.autotim_training import AutoTiMTrainer
from autotim.storage_client.file_store_manager import FileStoreManager, \
    DownloadFromStorageFailedError, StorageDoesNotExistError

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


def train_test_split(dataframe: pd.DataFrame, train_size):
    """Splits dataframe into train and test subset"""

    ids = dataframe[os.getenv("COLUMN_ID")].unique()
    np.random.shuffle(ids)
    train_ids = list(ids[:int(len(ids) * train_size)])
    train_df = dataframe[dataframe[os.getenv("COLUMN_ID")].isin(train_ids)]
    test_df = dataframe[~dataframe[os.getenv("COLUMN_ID")].isin(train_ids)]

    X_train = train_df.drop(columns=[os.getenv("COLUMN_LABEL")])
    y_train = get_labels(train_df)
    X_test = test_df.drop(columns=[os.getenv("COLUMN_LABEL")])
    y_test = get_labels(test_df)
    return X_train, y_train, X_test, y_test


def download_and_check_dataset(use_case_name: str, data_folder: str, bucket_dir: str,
                               client: FileStoreManager):
    dataset = None
    response = 'ok', status.HTTP_200_OK
    logging.debug(f"Downloading {bucket_dir} ...")

    try:
        client.download_dir(output_path=data_folder, prefix=bucket_dir)
        csv_files = glob.glob(
            os.path.join(data_folder, bucket_dir, '*.csv'))

        if len(csv_files) > 1:
            response = f"Your dataset directory '{bucket_dir}' contains multiple csv files " \
                       "that can be used for training your model." \
                       "This is an unexpected behavior. " \
                       "Please reduce the number of csv files in this directory to one. " \
                       "If you wish to store multiple datasets for your use case, please create" \
                       f"a new subdirectory in '{use_case_name}/' for each csv file or use our " \
                       f"/store-endpoint.", status.HTTP_406_NOT_ACCEPTABLE
        else:
            dataset = pd.read_csv(csv_files[0])

            for key in ['COLUMN_ID', 'COLUMN_LABEL', 'COLUMN_VALUE', 'COLUMN_KIND']:
                if os.getenv(key, "") != "" and not os.getenv(key) in dataset.columns:
                    response = f"Dataset does not contain the column '{os.getenv(key)}' " \
                               f"that was set as {key}", status.HTTP_406_NOT_ACCEPTABLE
    except StorageDoesNotExistError as e:
        response = "There has been an error connecting to storage: " + str(e), \
                   status.HTTP_404_NOT_FOUND
    except (DownloadFromStorageFailedError, FileNotFoundError) as e:
        response = "Could not load your requested dataset. Did you upload it already? " + \
                   str(e), status.HTTP_404_NOT_FOUND
    except (UnicodeDecodeError, pd.errors.ParserError, MemoryError) as e:
        response = "Could not read your dataset with pandas.read_csv: " + str(e), \
                   status.HTTP_406_NOT_ACCEPTABLE
    except AttributeError:
        response = "Dataset was not loaded correctly. Cannot proceed with training", \
                   status.HTTP_406_NOT_ACCEPTABLE

    return dataset, response


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


def dataset_split(name, data_folder, file_client, evaluation_identifier, dataset, train_size):
    if evaluation_identifier is not None:
        eval_data, eval_res = download_and_check_dataset(use_case_name=name,
                                                         data_folder=data_folder,
                                                         client=file_client,
                                                         bucket_dir=f"{name}/"
                                                                    f"{evaluation_identifier}/")
        if eval_res[1] != status.HTTP_200_OK or eval_data is None:
            raise DataSplitError(message=eval_res[0],status=eval_res[1])

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


def train(name: str, identifier: str, file_client: FileStoreManager, train_size,
          max_attempts, evaluation_identifier=None):
    """Run training for the given use case."""
    data_folder = os.path.join(str(Path(__file__).parent.parent), 'data')
    dataset, dataset_response = download_and_check_dataset(
        use_case_name=name,
        data_folder=data_folder,
        client=file_client,
        bucket_dir=f"{name}/{identifier}/")

    if dataset_response[1] != status.HTTP_200_OK:
        return jsonify({'training': "failed",
                        'error': dataset_response[0]}), dataset_response[1]

    try:
        x_train, x_test, y_train, y_test = \
            dataset_split(name=name, data_folder=data_folder, file_client=file_client,
                        evaluation_identifier=evaluation_identifier,
                        dataset=dataset, train_size=train_size)
    except DataSplitError as e:
        return jsonify({'training': "failed",
                            'error': e.message}), e.status

    # Extract Features
    try:
        features_train = extract_features(x_train, y_train)
    except FeatureCreationFailedError as e:
        return jsonify({'training': "failed",
                        'error': e.message}), status.HTTP_406_NOT_ACCEPTABLE

    warning = ''
    old_metric, new_metric = '', ''
    for attempt_model_creation in range(int(max_attempts)):

        features_train = select_relevant_features(features=features_train,
                                                  target_vector=y_train,
                                                  features_decrement_count=
                                                  attempt_model_creation)

        # Train Model
        try:
            experiment_name, model_version = train_model(
                name, identifier, features_train, y_train)
        except (H2OError, MlflowException) as e:
            logging.error(e)
            return jsonify({'training': "failed",
                            'error': "Internal error during training occurred."}), \
                status.HTTP_500_INTERNAL_SERVER_ERROR

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
                return jsonify({'training': "failed",
                                'error': "Internal error during training occurred." +
                                "Please define the paramater max_attempts " +
                                "from default '5' to an higer number " +
                                "in your request. For more information " +
                                "have a look into the README.md"}), \
                    status.HTTP_500_INTERNAL_SERVER_ERROR
            logging.warning(
                "Too many features for model prediction. Retrain model with less features")

    # Clear data folder
    if os.path.exists(data_folder):
        shutil.rmtree(data_folder)

    if old_metric != "" and new_metric != "" and old_metric is not None and new_metric is not None \
            and old_metric != new_metric:
        warning = f"The chosen metric has changed from {str(old_metric)} to " \
                  f"{str(new_metric)}. Changing metrics between experiment runs " \
                  f"is not encouraged. To guarantee that the desired model is set to " \
                  f"production, please do so manually."

    return jsonify(merge_response_dict(versions=model_version,
                                       warning=warning)), status.HTTP_200_OK
