"""Run AutoTiM training."""
import math
import tempfile
import json
import os
import logging
import tsfresh

import h2o
from h2o.exceptions import H2OServerError
from h2o.automl import H2OAutoML

import mlflow
from mlflow.tracking import MlflowClient


class AutoTiMTrainer:
    def __init__(self, experiment_name, model_name, tracking_uri=os.getenv('MLFLOW_TRACKING_URI', "http://localhost:5000")):
        try:
            h2o.init()
        except H2OServerError:
            logging.warning("H2O already initialized, move on training.")
        mlflow.set_tracking_uri(uri=tracking_uri)
        self.client = MlflowClient()
        self.experiment_name = experiment_name
        self.model_name = model_name

    def init_mlflow(self):
        if not mlflow.get_experiment_by_name(self.experiment_name):
            mlflow.create_experiment(name=self.experiment_name)
        experiment = mlflow.get_experiment_by_name(self.experiment_name).experiment_id
        return experiment

    @staticmethod
    def prepare_training_frame(data, labels):
        train = data.copy()
        train['label'] = labels
        train = h2o.H2OFrame(train).drop([0], axis=0)
        train['label'] = train['label'].asfactor() # labels are categorical
        x_cols = train.columns
        y_col = 'label'
        x_cols.remove(y_col)
        return train, x_cols, y_col

    def log(self, model, num_features, feature_extraction_settings):
        # Log model to MLFlow
        best_model = model.leader
        mlflow.h2o.log_model(best_model, self.model_name, registered_model_name=self.model_name)
        latest_version = self.client.get_latest_versions(self.model_name, stages=['None'])[-1] \
            .version
        self.client.transition_model_version_stage(name=self.model_name,
                                                   version=latest_version,
                                                   stage='Staging')
        # Log feature extraction settings to MLFlow
        mlflow.log_metric('number of extracted features', num_features)
        mlflow.log_params({
            'column_id': os.getenv("COLUMN_ID"),
            'column_value': os.getenv("COLUMN_VALUE"),
            'column_kind': os.getenv("COLUMN_KIND"),
            'column_sort': os.getenv("COLUMN_SORT")
        })
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
            json.dump(feature_extraction_settings, f)
            f.seek(0)
            mlflow.log_artifact(f.name, artifact_path='feature_extraction_settings')

        return latest_version

    def train(self, features, labels):
        experiment_id = self.init_mlflow()
        with mlflow.start_run(experiment_id=experiment_id):
            # AutoTiM Training
            training_frame, x, y = self.prepare_training_frame(features, labels)
            if os.getenv("TRAIN_TIME") == "dynamic":
                runtime = int(min(math.sqrt(training_frame.ncols * training_frame.nrows) + 120,
                                  1800))
                nmodels = 10
                aml = H2OAutoML(nfolds=5, max_runtime_secs_per_model=int(runtime / nmodels),
                                max_runtime_secs=runtime, keep_cross_validation_predictions=True,
                                max_models=nmodels,
                                stopping_tolerance=1 / runtime * math.sqrt(1 / nmodels))
                aml.train(x=x, y=y, training_frame=training_frame)
            else:
                aml = H2OAutoML(nfolds=5, max_runtime_secs=int(float(os.getenv("TRAIN_TIME"))*60),
                                keep_cross_validation_predictions=True)
                aml.train(x=x, y=y, training_frame=training_frame)

            # Log model and feature extraction settings to MLFlow
            num_features = len(features.columns)
            settings = tsfresh.feature_extraction.settings.from_columns(features)
            version = self.log(aml, num_features, settings)
        return version
