import io

from tests_autotim.app.app_test import AppTest, AUTH_HEADER

from autotim.storage_client.file_store_manager import StorageDoesNotExistError
from autotim.app.endpoints.utils.reponse_utils import RESPONSE_400_INPUT_FORMAT_WRONG, \
    RESPONSE_400_NO_REQUIRED_PARAM, RESPONSE_500_INTERNAL_SERVER_ERROR 
from autotim.app.endpoints.store_bp import RESPONSE_500_INTERNAL_SERVER_ERROR,\
    RESPONSE_400_BLOB_ALREADY_EXISTS


class StoreBPTest(AppTest):
    def test_store_returns_400_wrong_input_on_missing_file(self):
        response = self.client.post('/store', data={
                'some_irrelevant_param': 'possums are great'
            }, headers=AUTH_HEADER)

        self.assertEqual(response.status_code, RESPONSE_400_INPUT_FORMAT_WRONG.status_code)
        self.assertEqual(response.data, str.encode(RESPONSE_400_INPUT_FORMAT_WRONG.data.decode("utf-8")))

    def test_store_returns_400_wrong_input_on_multiple_files(self):
        response = self.client.post('/store', data={
                'file_1': (io.BytesIO(b"abcdef"), 'foo.csv'),
                'file_2': (io.BytesIO(b"abcdef"), 'bar.csv')
            }, headers=AUTH_HEADER)

        self.assertEqual(response.status_code, RESPONSE_400_INPUT_FORMAT_WRONG.status_code)
        self.assertEqual(response.data, str.encode(RESPONSE_400_INPUT_FORMAT_WRONG.data.decode("utf-8")))

    def test_store_returns_400_wrong_input_on_non_csv_file(self):
        response = self.client.post('/store', data={
                'file_1': (io.BytesIO(b"abcdef"), 'foo.txt')
            }, headers=AUTH_HEADER)

        self.assertEqual(response.status_code, RESPONSE_400_INPUT_FORMAT_WRONG.status_code)
        self.assertEqual(response.data, str.encode(RESPONSE_400_INPUT_FORMAT_WRONG.data.decode("utf-8")))

    def test_store_returns_400_no_required_param(self):
        response = self.client.post('/store', data={
                'file' : (io.BytesIO(b"abcdef"), 'foo.csv'),
                'some_irrelevant_param': 'possums are great'
            }, headers=AUTH_HEADER)

        self.assertEqual(response.status_code, RESPONSE_400_NO_REQUIRED_PARAM.status_code)
        self.assertEqual(response.data, str.encode(RESPONSE_400_NO_REQUIRED_PARAM.data.decode("utf-8")))

    def test_store_returns_400_blob_already_exists(self):
        self.file_client_mock.dir_exists.return_value = True

        response = self.client.post('/store', data={
                'file' : (io.BytesIO(b"abcdef"), 'foo.csv'),
                'use_case_name': 'possum', 'dataset_identifier': '42'
            }, headers=AUTH_HEADER)

        self.assertEqual(response.status_code, RESPONSE_400_BLOB_ALREADY_EXISTS.status_code)
        self.assertEqual(response.data, str.encode(RESPONSE_400_BLOB_ALREADY_EXISTS.data.decode("utf-8")))

    def test_store_returns_500_on_gcs_attribute_error(self):
        self.file_client_mock.dir_exists.side_effect = StorageDoesNotExistError('')

        response = self.client.post('/store', data={
                'file' : (io.BytesIO(b"abcdef"), 'foo.csv'),
                'use_case_name': 'possum', 'dataset_identifier': '42'
            }, headers=AUTH_HEADER)

        self.assertEqual(response.status_code, RESPONSE_500_INTERNAL_SERVER_ERROR.status_code)

    def test_store_returns_200(self):
        self.file_client_mock.dir_exists.return_value = False

        response = self.client.post('/store', data={
                'file' : (io.BytesIO(b"abcdef"), 'foo.csv'),
                'use_case_name': 'possum', 'dataset_identifier': '42'
            }, headers=AUTH_HEADER)

        self.assertEqual(response.status_code, 200)
