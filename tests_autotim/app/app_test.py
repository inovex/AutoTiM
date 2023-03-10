import unittest

from base64 import b64encode
from unittest.mock import MagicMock
from flask_injector import FlaskInjector
from injector import singleton

from autotim.app.app import app
from autotim.app.authconfig import AuthConfig
from autotim.storage_client.file_store_manager import FileStoreManager

from autotim.prediction_service.autotim_prediction_service import AutoTiMPredictionService


AUTH_HEADER = {
    'Authorization': 'Basic ' + b64encode(bytes(AuthConfig.AUTOTiM_USERNAME + ':' +
                                                AuthConfig.AUTOTiM_PASSWORD, "utf-8")
                                          ).decode('utf-8')
}


class AppTest(unittest.TestCase):
    def setUp(self) -> None:
        self.app = app
        self.app.config['TESTING'] = True

        self.file_client_mock = MagicMock(autospec=FileStoreManager)
        self.autotim_prediction_service = MagicMock(autospec=AutoTiMPredictionService)
        FlaskInjector(app=self.app, modules=[self.configure_mocks])

        self.client = self.app.test_client()

    def configure_mocks(self, binder):
        binder.bind(FileStoreManager, to=self.file_client_mock, scope=singleton)
        binder.bind(AutoTiMPredictionService, to=self.autotim_prediction_service,
                    scope=singleton)


class RouteTest(AppTest):
    def test_hello(self):
        response = self.client.get("/", headers=AUTH_HEADER)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, b'Hello, World!')


if __name__ == "__main__":
    unittest.main()
