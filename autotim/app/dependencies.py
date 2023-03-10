import os
from injector import singleton

from autotim.storage_client.file_store_manager import FileStoreManager
from autotim.storage_client.gcs_bucket_client import GCSBucketClient
from autotim.storage_client.local_client import LocalStorageClient

from autotim.prediction_service.autotim_prediction_service import AutoTiMPredictionService


def configure(binder):
    if os.getenv("STORAGE") == "local":
        binder.bind(FileStoreManager, to=LocalStorageClient, scope=singleton)
    else:
        os.environ["STORAGE"] = 'GCS'
        binder.bind(FileStoreManager, to=GCSBucketClient, scope=singleton)

    binder.bind(AutoTiMPredictionService,
                to=AutoTiMPredictionService, scope=singleton)
