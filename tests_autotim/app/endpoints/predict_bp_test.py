"""Prediction tests. """
import io

import pandas as pd
from h2o import H2OFrame
from mock import patch

from mlflow.exceptions import MlflowException
from requests.exceptions import ConnectionError

from autotim.model_selection.exceptions import MlflowModelNotFoundError
from autotim.feature_engineering.exceptions import FeatureCreationFailedError

from autotim.app.endpoints.utils.reponse_utils import RESPONSE_400_INPUT_FORMAT_WRONG, \
    RESPONSE_400_NO_REQUIRED_PARAM, RESPONSE_500_INTERNAL_SERVER_ERROR 
from autotim.app.endpoints.predict_bp import RESPONSE_404_MODEL_NOT_FOUND, \
     RESPONSE_400_FEATURES_NOT_CREATED

from tests_autotim.app.app_test import AppTest, AUTH_HEADER
from tests_autotim.test_objects.test_objects import set_environ_for_testing, \
get_correct_test_df_from_env


class PredictBPTest(AppTest):
    """Prediction tests. """

    def test_predict_returns_400_wrong_input_on_missing_file(self):
        response = self.client.post('/predict', data={
            'some_irrelevant_param': 'possums are great'
        }, headers=AUTH_HEADER)

        self.assertEqual(response.status_code, RESPONSE_400_INPUT_FORMAT_WRONG.status_code)
        self.assertEqual(response.data, str.encode(RESPONSE_400_INPUT_FORMAT_WRONG.data.decode("utf-8")))

    def test_predict_returns_400_wrong_input_on_multiple_files(self):
        response = self.client.post('/predict', data={
            'file_1': (io.BytesIO(b"abcdef"), 'foo.csv'),
            'file_2': (io.BytesIO(b"abcdef"), 'bar.json')
        }, headers=AUTH_HEADER)

        self.assertEqual(response.status_code, RESPONSE_400_INPUT_FORMAT_WRONG.status_code)
        self.assertEqual(response.data, str.encode(RESPONSE_400_INPUT_FORMAT_WRONG.data.decode("utf-8")))

    def test_predict_returns_400_wrong_input_on_wrong_file_extension(self):
        response = self.client.post('/predict', data={
            'file_1': (io.BytesIO(b"abcdef"), 'foo.zip')
        }, headers=AUTH_HEADER)

        self.assertEqual(response.status_code, RESPONSE_400_INPUT_FORMAT_WRONG.status_code)
        self.assertEqual(response.data, str.encode(RESPONSE_400_INPUT_FORMAT_WRONG.data.decode("utf-8")))

    def test_predict_returns_400_no_required_param(self):
        response = self.client.post('/predict', data={
            'file': (io.BytesIO(b"abcdef"), 'foo.csv'),
            'some thing': 'another thing'
        }, headers=AUTH_HEADER)

        self.assertEqual(response.status_code, RESPONSE_400_NO_REQUIRED_PARAM.status_code)
        self.assertEqual(response.data, str.encode(RESPONSE_400_NO_REQUIRED_PARAM.data.decode("utf-8")))

    @patch('autotim.app.endpoints.predict_bp.create_features',
           side_effect=FeatureCreationFailedError(message=''))
    def test_predict_returns_generic_400_features_not_created(self, *_):
        response = self.client.post('/predict', data={
            'file': (io.BytesIO(b"abcdef"), 'foo.csv'),
            'use_case_name': 'some_name',
            'dataset_identifier': 'some identifier'
        }, headers=AUTH_HEADER)

        self.assertEqual(response.status_code, RESPONSE_400_FEATURES_NOT_CREATED.status_code)

    def test_predict_returns_400_features_not_created_on_empty_df(self):
        # when dataframe is empty
        self.autotim_prediction_service.read_timeseries_from_file.return_value = pd.DataFrame()

        response = self.client.post('/predict', data={
            'file': (io.BytesIO(b"abcdef"), 'foo.csv'),
            'use_case_name': 'some_name',
            'dataset_identifier': 'some identifier'
        }, headers=AUTH_HEADER)

        self.assertEqual(response.status_code, RESPONSE_400_FEATURES_NOT_CREATED.status_code)

    def test_predict_returns_400_features_not_created_on_missing_settings(self):
        # when settings are None
        self.autotim_prediction_service.load_settings_for_model.return_value = None
        # when dataframe is correct
        set_environ_for_testing(env_vars=['COLUMN_ID', 'COLUMN_VALUE', 'COLUMN_KIND'],
                                set_with_random=True)
        self.autotim_prediction_service.read_timeseries_from_file = \
            get_correct_test_df_from_env()

        response = self.client.post('/predict', data={
            'file': (io.BytesIO(b"abcdef"), 'foo.csv'),
            'use_case_name': 'some_name',
            'dataset_identifier': 'some identifier'
        }, headers=AUTH_HEADER)

        self.assertEqual(response.status_code, RESPONSE_400_FEATURES_NOT_CREATED.status_code)

    def test_predict_returns_400_features_not_created_on_empty_settings(self):
        # when settings are None
        self.autotim_prediction_service.load_settings_for_model.return_value = {}
        # when dataframe is correct
        set_environ_for_testing(env_vars=['COLUMN_ID', 'COLUMN_VALUE', 'COLUMN_KIND'],
                                set_with_random=True)
        self.autotim_prediction_service.read_timeseries_from_file = \
            get_correct_test_df_from_env()

        response = self.client.post('/predict', data={
            'file': (io.BytesIO(b"abcdef"), 'foo.csv'),
            'use_case_name': 'some_name',
            'dataset_identifier': 'some identifier'
        }, headers=AUTH_HEADER)

        self.assertEqual(response.status_code, RESPONSE_400_FEATURES_NOT_CREATED.status_code)

    def test_predict_returns_404_model_not_found(self):
        # when settings cannot be loaded, because a model was not found
        self.autotim_prediction_service.get_model.side_effect = \
            MlflowModelNotFoundError(model_name='')

        response = self.client.post('/predict', data={
            'file': (io.BytesIO(b"abcdef"), 'foo.csv'),
            'use_case_name': 'some_name',
            'dataset_identifier': 'some identifier'
        }, headers=AUTH_HEADER)

        self.assertEqual(response.status_code, RESPONSE_404_MODEL_NOT_FOUND.status_code)
        self.assertEqual(response.data, str.encode(RESPONSE_404_MODEL_NOT_FOUND.data.decode("utf-8")))

    def test_predict_returns_500_on_connection_error(self):
        self.autotim_prediction_service.get_model.side_effect = ConnectionError

        response = self.client.post('/predict', data={
            'file': (io.BytesIO(b"abcdef"), 'foo.csv'),
            'use_case_name': 'some_name',
            'dataset_identifier': 'some identifier'
        }, headers=AUTH_HEADER)

        self.assertEqual(response.status_code, RESPONSE_500_INTERNAL_SERVER_ERROR.status_code)
        self.assertEqual(response.data, str.encode(RESPONSE_500_INTERNAL_SERVER_ERROR.data.decode("utf-8")))

    @patch('autotim.app.endpoints.predict_bp.create_features',
           return_value=H2OFrame())
    def test_predict_returns_500_on_mlflow_error(self, *_):
        self.autotim_prediction_service.predict.side_effect = MlflowException('')

        response = self.client.post('/predict', data={
            'file': (io.BytesIO(b"abcdef"), 'foo.csv'),
            'use_case_name': 'some_name',
            'dataset_identifier': 'some identifier'
        }, headers=AUTH_HEADER)
        self.assertEqual(response.status_code, RESPONSE_500_INTERNAL_SERVER_ERROR.status_code)
        self.assertEqual(response.data, str.encode(RESPONSE_500_INTERNAL_SERVER_ERROR.data.decode("utf-8")))

    @patch('autotim.app.endpoints.predict_bp.create_features', return_value=H2OFrame())
    def test_predict_returns_200(self, *_):
        self.autotim_prediction_service.predict.return_value = [1]

        response = self.client.post('/predict', data={
            'file': (io.BytesIO(b"abcdef"), 'foo.csv'),
            'use_case_name': 'some_name',
            'dataset_identifier': 'some identifier'
        }, headers=AUTH_HEADER)

        self.assertEqual(response.status_code, 200)
