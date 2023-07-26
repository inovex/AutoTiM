import os
import logging
import h2o
import mlflow
from h2o.exceptions import H2OServerError

from injector import inject

from autotim.app.endpoints.utils.dataframe_utils import convert_h2oframe_to_numeric

from autotim.prediction_service.autotim_model import AutoTiM_Model
from autotim.prediction_service.mlflow_model_loader import MlFlowModelLoader


mlflow.set_tracking_uri(uri=os.getenv('MLFLOW_TRACKING_URI', "http://localhost:5000"))


class AutoTiMPredictionService:
    current_model: AutoTiM_Model = None

    @inject
    def __init__(self):
        try:
            h2o.init()
        except H2OServerError:
            logging.warning("H2O already initialized, move on prediction.")
        self.client = mlflow.tracking.MlflowClient()

    @staticmethod
    def predict(model, features):
        features = convert_h2oframe_to_numeric(features, features.columns)

        import sys
        sys.setrecursionlimit(10000)  # got an recursion error locally

        result = model.predict(features)
        sys.setrecursionlimit(999)

        return result.as_data_frame()['predict'].tolist()

    def get_model(self, use_case_name: str, dataset_identifier: str,
                  model_version: int = None) -> AutoTiM_Model:
        stage = None if model_version is not None else 'Production'

        model_sceleton = AutoTiM_Model(use_case_name=use_case_name,
                                      dataset_identifier=dataset_identifier,
                                      model_version=model_version, stage=stage)
        return self._get_model(model_sceleton=model_sceleton)

    def _get_model(self, model_sceleton: AutoTiM_Model) -> AutoTiM_Model:
        """
        Helper function to facilitate storage of the last used model
        (useful if one particular model is in use through multiple consecutive requests,
        so that it can be loaded from MLFlow only once for the 1st request).
        """
        current_model_uri = self.current_model.mlflow_uri \
            if self.current_model is not None else None
        if current_model_uri != model_sceleton.mlflow_uri:
            loader = MlFlowModelLoader(mlflow_client=self.client)
            self.current_model = loader.retrieve_mlflow_model_data(
                model_sceleton=model_sceleton
            )
        return self.current_model
