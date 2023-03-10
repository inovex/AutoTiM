import os
import tempfile

from datetime import datetime
from flask import Blueprint, request, Response, jsonify
from flask_api import status
from injector import inject

from autotim.storage_client.file_store_manager import FileStoreManager, \
    StorageDoesNotExistError, UploadToStorageFailedError
from autotim.app.endpoints.utils.reponse_utils import RESPONSE_400_INPUT_FORMAT_WRONG, \
    RESPONSE_400_NO_REQUIRED_PARAM, RESPONSE_500_INTERNAL_SERVER_ERROR

from autotim.app.endpoints.utils.file_handling_utils import get_single_file_input_format

STORE_BP = Blueprint('store', __name__)

RESPONSE_400_BLOB_ALREADY_EXISTS = Response("Dataset with this name and version already exists. "
                                            "Please choose a new dataset version identifier or "
                                            "remove the existing data from storage.",
                                            status=400)


@inject
@STORE_BP.route('/store', methods=['POST'])
def store(file_client: FileStoreManager):
    """
    Stores data (only accepts a single .csv-file) in a gcs bucket
        specified through an environment variable or in gcs_config.py.
    Required parameters: use_case_name (str), file (.csv-file).
    Optional parameters: dataset_identifier (str).

    Data will be stored under a remote path: <use_case_name> / <dataset_identifier>.
    """
    # check file input
    file = get_single_file_input_format(request=request, allowed_extensions=['csv'])
    if file is None:
        return RESPONSE_400_INPUT_FORMAT_WRONG

    use_case_name = request.form.get('use_case_name', None)
    if use_case_name is None:
        return RESPONSE_400_NO_REQUIRED_PARAM

    # either a dataset_identifier is provided, else: use timestamp of upload as version
    dataset_identifier = request.form.get('dataset_identifier') \
        if request.form.get('dataset_identifier') is not None \
        else datetime.now().strftime("%Y-%b-%d-%H:%M")

    # upload data to gcs bucket
    try:
        # stop upload if dataset_identifier and dataset_identifier combination already in use
        dest_blob = os.path.join(use_case_name, dataset_identifier)
        if file_client.dir_exists(dest_blob):
            return RESPONSE_400_BLOB_ALREADY_EXISTS

        with tempfile.TemporaryDirectory() as tdir:
            temp_path = os.path.join(tdir, file.filename)
            file.save(temp_path)
            file_client.save_single_file(src=temp_path, dest=dest_blob)
    except (StorageDoesNotExistError, UploadToStorageFailedError):
        return RESPONSE_500_INTERNAL_SERVER_ERROR

    return jsonify({'use_case_name': use_case_name,
                    'dataset_identifier': dataset_identifier}), \
           status.HTTP_200_OK
