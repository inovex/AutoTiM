import os
import json
import logging

from pathlib import Path


class Config:
    GOOGLE_CLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT', None)
    GOOGLE_CLOUD_BUCKET = os.getenv('GOOGLE_CLOUD_BUCKET', "preprocessed-usecase-data")
    GOOGLE_CLOUD_CREDENTIALS = os.getenv('GOOGLE_CLOUD_CREDENTIALS', None)

    if os.getenv('STORAGE') != 'local':
        if GOOGLE_CLOUD_PROJECT is None:
            logging.debug('GOOGLE_CLOUD_PROJECT not set, parsing from INSTANCE_CONNECTION_NAME.')
            try:
                GOOGLE_CLOUD_PROJECT = os.getenv('INSTANCE_CONNECTION_NAME', '').split(':', 1)[0]
            except (AttributeError, KeyError) as e:
                logging.error(f"GOOGLE_CLOUD_PROJECT not set: {e}")

        if GOOGLE_CLOUD_CREDENTIALS is None:
            # read gcs credentials from a local file
            if os.getenv('GCP_CREDENTIALS_PATH') is None:
                os.environ["GCP_CREDENTIALS_PATH"] = os.path.join(Path(__file__).parent.parent,
                                                                  'gcloud-cicd-service-key.json')
            try:
                with open(os.getenv('GCP_CREDENTIALS_PATH'), 'r', encoding='utf8') as f:
                    GOOGLE_CLOUD_CREDENTIALS = json.load(f)
            except (TypeError, FileNotFoundError) as e:
                logging.error(f"GOOGLE_CLOUD_CREDENTIALS not found: {e}")
    elif os.getenv('STORAGE') == 'local' and not (os.getenv("GCP_CREDENTIALS_PATH") is None or
                                                  os.getenv("GCP_CREDENTIALS_PATH") == ""):
        logging.error("Your STORAGE is set to 'local' and you have provided a "
                      "GCP_CREDENTIALS_PATH: this is an unexpected behavior. "
                      "We will use 'local' STORAGE for now, please rerun with "
                      "STORAGE set to 'GCS' to access Google Cloud.")
