"""Connect to Google Cloud Bucket."""
import os
import glob
import logging
import shutil

import gcsfs

from injector import inject
from google.auth import exceptions as auth_exception
from google.cloud.storage import Client, Bucket
from google.oauth2 import service_account
from google.api_core.exceptions import BadRequest, Forbidden

from autotim.storage_client.file_store_manager import FileStoreManager, \
    StorageDoesNotExistError, DownloadFromStorageFailedError
from autotim.storage_client.gcs_storage_config import Config


class GCSBucketClient(FileStoreManager):
    @inject
    def __init__(self, bucket_name: str = ""):
        FileStoreManager.__init__(self=self)
        try:
            credentials = service_account.Credentials.from_service_account_info(
                Config.GOOGLE_CLOUD_CREDENTIALS)
            self._gcs_client = Client(project=Config.GOOGLE_CLOUD_PROJECT,
                                      credentials=credentials)

            self._bucket_name = Config.GOOGLE_CLOUD_BUCKET \
                if (bucket_name is None or bucket_name == "") \
                else bucket_name
            self._bucket = self._init_bucket_client(self._bucket_name)
        except (KeyError, AttributeError, BucketNotFoundException) as e:
            logging.error(f'Connection to GCSBucket was not established: {e}')

    def dir_exists(self, dir_path: str):
        # gc does not have a notion of a "folder" -> this is a custom check if a folder exists
        try:
            blobs = self._bucket.list_blobs()
            return any(blob.name.startswith(dir_path) for blob in blobs)
        except AttributeError as ae:
            raise BucketNotFoundException(bucket_name=self._bucket_name) from ae

    def save_single_file(self, src: str, dest: str):
        try:
            remote_path = f'{dest}/{"/".join(src.split(os.sep)[-1:])}'
            if os.path.isfile(src):
                blob = self._bucket.blob(remote_path)
                blob.upload_from_filename(src)
        except AttributeError as ae:
            raise BucketNotFoundException(bucket_name=self._bucket_name) from ae

    def upload_files_from_local_dir(self, local_path: str, dest_blob_name: str):
        rel_paths = glob.glob(local_path + '/**', recursive=True)
        for local_file in rel_paths:
            if os.path.isfile(local_file):
                self.save_single_file(src=local_file, dest=dest_blob_name)

    def download_dir(self, output_path, prefix=""):
        if not self.dir_exists(prefix):
            logging.warning(f"Path '{prefix}' in bucket '"
                            f"{self._bucket_name}' does not exist.")
            raise DownloadFromStorageFailedError(volume_name=self._bucket_name,
                                                 prefix=prefix)

        if os.path.exists(output_path):
            # if path to download already exists -> delete directory
                # so that new data can be downloaded
            shutil.rmtree(output_path)
        else: # create path to download files into
            os.makedirs(output_path)

        # Get list of files
        blobs = self._bucket.list_blobs(prefix=prefix)
        for blob in blobs:
            file_path = os.path.join(output_path, blob.name)
            dir_name = os.path.dirname(file_path)
            if file_path[-1] == '/':
                continue
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            # Download
            blob.download_to_filename(file_path)

    def _blob_exists(self, blob_name: str) -> bool:
        blob = self._bucket.blob(blob_name)
        return blob.exists()

    def _init_bucket_client(self, bucket_name: str) -> Bucket:
        if not self._bucket_exists(bucket_name):
            logging.error(f'Bucket {bucket_name} was not found.')
            raise BucketNotFoundException(bucket_name=bucket_name)
        self._bucket_name = bucket_name
        return self._gcs_client.get_bucket(bucket_name)

    def get_gcs_filesystem(self):
        """
        :return: GCSFileSystem ready to interact with the project's storage bucket on gcs
        """

        try:
            gcs_filesystem = gcsfs.GCSFileSystem(project=os.getenv('GOOGLE_CLOUD_PROJECT'),
                                                 token=self._gcs_client)
            return gcs_filesystem
        except auth_exception.DefaultCredentialsError:
            logging.error("Default credentials not found.")
        return None

    def download_single_blob(self, output_path, gcs_blob_name):
        """
        Download the specified file from the specified google-cloud
         directory into the specified local directory.
        :param gcs_blob_name: path to the file in google-cloud that will be downloaded
        :param output_path: local destination path to the file that will be downloaded
        """

        if not self._blob_exists(gcs_blob_name):
            raise BlobNotFoundException(bucket_name=self._bucket_name,
                                        blob_name=gcs_blob_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        blob = self._bucket.get_blob(gcs_blob_name)
        file_path = os.path.join(output_path, blob.name)
        dir_name = os.path.dirname(file_path)
        if file_path[-1] == '/':
            logging.warning("The blob requested for download is a folder. ")
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        # Download
        blob.download_to_filename(file_path)

    def _bucket_exists(self, bucket_name: str) -> bool:
        try:
            return self._gcs_client.lookup_bucket(bucket_name) is not None
        except (IndexError, KeyError, BadRequest, Forbidden):
            return False


class BlobNotFoundException(Exception):
    def __init__(self, bucket_name: str, blob_name: str):
        self.message = f"The blob '{blob_name}' was not found in {bucket_name}."
        super().__init__(self.message)


class BucketNotFoundException(StorageDoesNotExistError):
    def __init__(self, bucket_name: str):
        self.message = f"The bucket '{bucket_name}' was not found or does not exist. " \
                       f"Consider checking the bucket name syntax."
        super().__init__(self.message)
