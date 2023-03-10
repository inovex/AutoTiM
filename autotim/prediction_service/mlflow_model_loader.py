"""This class facilitates communication with MlFlow and retrieval of stored models."""
import os
import json
import tempfile
import logging

import mlflow
from mlflow.exceptions import RestException, MlflowException

from autotim.prediction_service.autotim_model import AutoTiM_Model
from autotim.model_selection.exceptions import MlflowModelNotFoundError, \
    ModelArtifactsNotAvailableError


class MlFlowModelLoader:

    def __init__(self, mlflow_client):
        self.client = mlflow_client

    def get_model_run_id_and_version(self, model_name: str, model_version: int = None,
                                     stage: str = 'Production'):
        """
        Returns model run id from a specified model version.
        If no version is provided, a model set to a production stage will be used.
        """
        if model_version is not None:
            return self.client.get_model_version(
                name=model_name, version=model_version).run_id, model_version
        # if model version not provided -> pull a model from production
        latest = self.client.get_latest_versions(model_name, stages=[stage])[-1]
        return latest.run_id, latest.version

    def get_params_for_run_id(self, run_id: str, param_keys):
        """
        Retrieves parameters associated with a run_id.

        Intended use: parameters needed to further use of a model stored within a Run,
            were persisted by the AutoTiM-service as Run-params. Use this function
            to retrieve these params and reuse them in (e.g.) prediction with a model.
        """
        all_run_id_params = self.client.get_run(run_id).data.params
        result_params = {}

        for key in param_keys:
            if all_run_id_params.get(key) == "":
                # mlflow does not store None-values, so we substitute these with an empty str
                result_params[key] = None
            else:
                result_params[key] = all_run_id_params.get(key, None)

        return result_params

    def load_settings_for_run_id(self, run_id):
        """Load the settings for the given run id."""
        with tempfile.TemporaryDirectory() as t_dir:
            path = self.client.download_artifacts(run_id, 'feature_extraction_settings/', t_dir)
            setting_file = os.path.join(path, os.listdir(path)[0])
            # pylint: disable=bad-option-value,unspecified-encoding
            with open(setting_file) as settings:
                extraction_settings = json.load(settings)
        return extraction_settings

    def retrieve_mlflow_model_data(self, model_sceleton: AutoTiM_Model):
        """
        Fills autotim_model instance with params, that need to be retrieved from MlFlow:
            - model run id
            - model version (if not set already)
            - h20 model
            - feature engineering settings
            - model_params: 'column_id', 'column_value', 'column_kind', 'column_sort'

        :param model_sceleton base for the AutoTiM_Model containing a model name and uri
        :return AutoTiM_Model
        """
        try:
            logging.debug(f"Loading model from MlFlow: {model_sceleton.autotim_model_name}")
            run_id, version = self.get_model_run_id_and_version(
                model_name=model_sceleton.autotim_model_name,
                model_version=model_sceleton.model_version,
                stage=model_sceleton.stage
            )
            model_sceleton.run_id = run_id
            model_sceleton.model_version = version

            model_sceleton.params = self.get_params_for_run_id(run_id=run_id,
                param_keys=['column_id', 'column_value', 'column_kind', 'column_sort'])

            model_sceleton.model = mlflow.h2o.load_model(model_uri=model_sceleton.mlflow_uri)
            model_sceleton.feature_settings = self.load_settings_for_run_id(run_id=run_id)

            return model_sceleton
        except (AttributeError, RestException, MlflowException) as e:
            # IndexError, AttributeError -> happens when asking for a model that does not exist
                # e.g. indexing the last model in an array of production models, but array is empty
            # RestException, MlflowException -> mlflow connection error
            logging.error(e)
            raise MlflowModelNotFoundError(model_name=model_sceleton.autotim_model_name) from e

        except (IndexError, FileNotFoundError, IOError, OSError) as e:
            # Cannot load setting or checkpoints
            logging.error(e)
            if model_sceleton.run_id is not None:
                # if run_id exists (aka model has metadata saved in the database),
                    # but the actual file containing checkpoints or feature settings is not found
                raise ModelArtifactsNotAvailableError(model_name=model_sceleton.autotim_model_name,
                            model_version=model_sceleton.model_version) from e
            raise MlflowModelNotFoundError(model_name=model_sceleton.autotim_model_name) from e
