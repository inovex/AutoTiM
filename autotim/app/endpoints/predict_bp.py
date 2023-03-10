import logging

from flask_api import status
from injector import inject

from flask import Blueprint, request, Response, jsonify
from mlflow.exceptions import MlflowException
from requests.exceptions import ConnectionError

from autotim.app.endpoints.utils.reponse_utils import RESPONSE_400_INPUT_FORMAT_WRONG, \
    RESPONSE_400_NO_REQUIRED_PARAM, RESPONSE_500_INTERNAL_SERVER_ERROR

from autotim.app.endpoints.utils.file_handling_utils import read_timeseries_from_file, \
     get_single_file_input_format

from autotim.prediction_service.autotim_prediction_service import AutoTiMPredictionService
from autotim.model_selection.exceptions import MlflowModelNotFoundError, \
    ModelArtifactsNotAvailableError

from autotim.feature_engineering.automated_feature_engineering import create_features
from autotim.feature_engineering.exceptions import FeatureCreationFailedError

PREDICT_BP = Blueprint('predict', __name__)

RESPONSE_404_MODEL_NOT_FOUND = \
    Response("Model not found: "
             "Your input for model name and version did not result in a valid combination. "
             "If you have only provided a name but no version,"
             " no model in Production stage was found",
             status=404)

RESPONSE_400_FEATURES_NOT_CREATED = Response(status=400)


@inject
@PREDICT_BP.route('/predict', methods=['POST'])
def predict(autotim_prediction_service: AutoTiMPredictionService):
    """
    Performs classification of a timeseries provided in a file-parameter
        with a model trained and stored with MLFlow.
    Required parameters: use_case_name (str),
                         dataset_identifier (str),
                         file (.csv- or .json-file).
    Optional parameters: model_version (str).

    Array of predictions will be returned in a response body.
    """
    timeseries = read_timeseries_from_file(
        file=get_single_file_input_format(request=request,
                                          allowed_extensions=['csv', 'json']))
    if timeseries is None:
        return RESPONSE_400_INPUT_FORMAT_WRONG

    if request.form.get('use_case_name') is None or \
            request.form.get('dataset_identifier') is None:
        return RESPONSE_400_NO_REQUIRED_PARAM

    use_case_name = request.form.get('use_case_name')
    dataset_identifier = request.form.get('dataset_identifier')
    model_version = request.form.get('model_version', None)

    try:
        autotim_model = autotim_prediction_service.get_model(use_case_name=use_case_name,
                                                           dataset_identifier=dataset_identifier,
                                                           model_version=model_version)

        features = create_features(dataframe=timeseries, settings=autotim_model.feature_settings,
                                   column_id=autotim_model.params.get('column_id'),
                                   column_value=autotim_model.params.get('column_value'),
                                   column_kind=autotim_model.params.get('column_kind'),
                                   column_sort=autotim_model.params.get('column_sort'))

        prediction = autotim_prediction_service.predict(features=features,
                                                       model=autotim_model.model)

    except FeatureCreationFailedError as e:
        return Response(e.message, status=RESPONSE_400_FEATURES_NOT_CREATED.status)
    except (MlflowModelNotFoundError, ModelArtifactsNotAvailableError):
        return RESPONSE_404_MODEL_NOT_FOUND
    except (MlflowException, ConnectionError, KeyError, RecursionError) as e:
        # KeyError -> environment variables not set
        logging.error(e)
        return RESPONSE_500_INTERNAL_SERVER_ERROR

    return jsonify({'prediction': prediction}), status.HTTP_200_OK
