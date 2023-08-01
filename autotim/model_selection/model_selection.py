"""Select models to set to production."""
import logging
import math
import os
import mlflow
import numpy
from pprint import pprint
from matplotlib import pyplot as plt
from pypdf import PdfMerger

from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, \
    accuracy_score, confusion_matrix

from autotim.feature_engineering.automated_feature_engineering import create_features
from autotim.app.endpoints.utils.dataframe_utils import convert_h2oframe_to_numeric

from autotim.prediction_service.autotim_model import AutoTiM_Model
from autotim.prediction_service.mlflow_model_loader import MlFlowModelLoader

from autotim.model_selection.exceptions import MlflowExperimentNotFoundError, \
    MlflowModelNotFoundError, ModelSelectionFailed, ModelArtifactsNotAvailableError
from autotim.model_selection.utils import is_better
from autotim.model_selection.pdf_creation import MetricsReportPDF

VALID_METRICS = ['accuracy',
                 'balanced_accuracy',
                 'recall_score',
                 'precision_score']


def create_plot(y_test, conf_matrix):
    unique_labels = numpy.unique(y_test)
    fig_size = 2 * math.sqrt(len(unique_labels) + 2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    ax.matshow(conf_matrix, cmap='viridis')
    ax.set_xticks(numpy.arange(len(unique_labels)), labels=unique_labels)
    ax.set_yticks(numpy.arange(len(unique_labels)), labels=unique_labels)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

    plt.xticks(fontsize=10, rotation=50)
    plt.yticks(fontsize=10)
    plt.xlabel('Predictions', fontsize=5 * math.sqrt(len(unique_labels)) + 5)
    plt.ylabel('Actual', fontsize=5 * math.sqrt(len(unique_labels)) + 5)
    plt.title('Confusion Matrix', fontsize=5 * math.sqrt(len(unique_labels)) + 5)
    plt.tight_layout()
    return fig


class ModelSelector:
    """Organize MlFlow Model stages."""
    latest_model: AutoTiM_Model = None
    production_model: AutoTiM_Model = None
    production_artifact_unavailable = False # whether prod model has to be set back from Production
    reset_prod_model_version = None # save production model version for the stage reset later

    def __init__(self, name, identifier, latest_model_version):
        mlflow.set_tracking_uri(uri=os.getenv('MLFLOW_TRACKING_URI', "http://localhost:5000"))
        self.client = mlflow.tracking.MlflowClient()

        self.experiment_name = name + '-' + identifier
        self.model_name = name + '-' + identifier + '_model'
        if not mlflow.get_experiment_by_name(self.experiment_name):
            raise MlflowExperimentNotFoundError(self.experiment_name)

        # load the latest trained model and the production model
        self._load_models_for_comparison(name=name, identifier=identifier,
                                         latest_model_version=latest_model_version)

    def compute_metrics(self, autotim_model: AutoTiM_Model, x_test, y_test) -> dict:
        """Computes and logs metrics for the latest model in the given stage,
         shows confusion matrix."""
        if autotim_model.model is None:
            return {}

        features = create_features(x_test, column_id=os.getenv("COLUMN_ID"),
                                   column_value=os.getenv("COLUMN_VALUE"),
                                   column_kind=os.getenv("COLUMN_KIND"),
                                   column_sort=os.getenv("COLUMN_SORT"),
                                   settings=autotim_model.feature_settings)

        features = convert_h2oframe_to_numeric(features, features.columns)
        y_pre = autotim_model.model.predict(features)['predict'].as_data_frame()
        num_features = self.client.get_metric_history(run_id=autotim_model.run_id, key="number of extracted features")
        num_features = int(num_features[0].value)

        run = mlflow.get_run(run_id=autotim_model.run_id)
        params = run.data.params
        num_train_data = params.get('number_of_training_instances')

        metrics = {
            'accuracy': accuracy_score(y_test, y_pre),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pre),
            'precision_score': precision_score(y_test, y_pre,
                                               average=os.getenv("RECALL_AVERAGE")),
            'recall_score': recall_score(y_test, y_pre, average=os.getenv("RECALL_AVERAGE"))
        }

        conf_matrix = confusion_matrix(y_test, y_pre)
        self.client.log_dict(run_id=autotim_model.run_id,
                             dictionary=numpy.array(conf_matrix).tolist(),
                             artifact_file="confusion_matrix.json")

        fig = create_plot(y_test=y_test, conf_matrix=conf_matrix)
        self.client.log_figure(run_id=autotim_model.run_id,
                               figure=fig, artifact_file="confusion_matrix.png")

        # log the metrics to mlflow
        for key in metrics:
            self.client.log_metric(
                run_id=autotim_model.run_id, key=key, value=metrics[key])

        # create pdf
        pdf_file_path = f"model_metrics_report.pdf"
        logging.info(f"Write PDF file to {pdf_file_path}")
        pdf = MetricsReportPDF(metrics, autotim_model.autotim_model_name)
        pdf.set_general_components()
        pdf.create_table_of_content()
        pdf.create_train_info_table(num_train_data, num_features)
        pdf.create_metrics_table()
        pdf.integrate_confusion_matrix(fig)
        pdf.output(pdf_file_path)
        # merge report with inovex description
        pdfs = [pdf_file_path,
                     os.path.join(os.path.dirname(__file__), "report_data/inovex_marketing_description.pdf")]

        merger = PdfMerger()

        for pdf in pdfs:
            merger.append(pdf)

        merged_pdf_path = f"{os.environ['data_folder']}/outputs/inovex_report.pdf"
        merger.write(merged_pdf_path)
        merger.close()
        self.client.log_artifact(
            run_id=autotim_model.run_id, local_path=merged_pdf_path, artifact_path="model_statistics")

        return metrics

    def metric_has_changed(self, metric: str) -> (bool, str):
        """
        Returns true if metric has changed.
        If so, it also returns the previously used metric
        """
        if self.production_model is not None:
            #  a production model exists.
            run = mlflow.get_run(run_id=self.production_model.run_id)
            try:
                prod_evaluation_metric = run.data.params['evaluation_metric']
            except KeyError:
                prod_evaluation_metric = None

            if prod_evaluation_metric is not None and prod_evaluation_metric != metric:
                return prod_evaluation_metric
        return None

    def update_production_flag(self, x_test, y_test, metric: str = 'accuracy'):
        """
        Compare the latest model that has been set to staging and the current production model.
        According to the given metric, give the production flag to the better model.
        """
        #  calculate the metrics for the latest model and the current production model
        if self.production_model is not None:
            metrics_production = self.compute_metrics(autotim_model=self.production_model,
                                                      x_test=x_test, y_test=y_test)
        else:
            metrics_production = {}

        metrics_latest = self.compute_metrics(autotim_model=self.latest_model,
                                              x_test=x_test, y_test=y_test)

        #  log the parameter that the model is being selected by
        self.client.log_param(run_id=self.latest_model.run_id,
                              key='evaluation_metric', value=metric)

        #  model selection logic
        set_latest_to_prod = False
        reset_production_model = False
        if not metrics_production or self.production_model is None:
            # there is currently no production model. Set the newest model to production
            set_latest_to_prod = True
        else:
            # production model metrics have been successfully calculated, compare to latest model.
            if metric not in VALID_METRICS:
                #  use default metric (accuracy)
                metric = 'accuracy'
            #  provided metric is valid or default -> set model to production if it is better
            if is_better(metric, metrics_latest[metric], metrics_production[metric]):
                set_latest_to_prod = True
                reset_production_model = True

        model_v_return = {'latest_model version' : self.latest_model.model_version,
                          'latest_model '+metric : metrics_latest[metric]}
        if set_latest_to_prod:
            self.client.transition_model_version_stage(name=self.model_name, stage='Production',
                                                       version=self.latest_model.model_version)
            model_v_return['best_model / production_model version'] = \
                self.latest_model.model_version
        else:
            model_v_return['best_model / production_model version'] = \
                self.production_model.model_version
            model_v_return['production_model '+metric] = metrics_production[metric]

        if reset_production_model or self.production_artifact_unavailable:

            model_version = self.production_model.model_version \
                if self.production_model else self.reset_prod_model_version

            if model_version is not None:
                self.client.transition_model_version_stage(name=self.model_name, stage='Staging',
                                                           version=model_version)

        return model_v_return

    def _load_models_for_comparison(self, name, identifier, latest_model_version):
        loader = MlFlowModelLoader(mlflow_client=self.client)

        try:
            self.latest_model = loader.retrieve_mlflow_model_data(
                model_sceleton=AutoTiM_Model(use_case_name=name, dataset_identifier=identifier,
                                         model_version=latest_model_version))
        except (MlflowModelNotFoundError, ModelArtifactsNotAvailableError) as e:
            raise ModelSelectionFailed(message='Latest model not available: '
                                               'nothing to select.') from e

        if int(latest_model_version) > 1:
            try:
                self.production_model = loader.retrieve_mlflow_model_data(
                    model_sceleton=AutoTiM_Model(use_case_name=name, dataset_identifier=identifier,
                                                 model_version=None, stage='Production'))
            except (MlflowModelNotFoundError, ModelArtifactsNotAvailableError) as e:
                if isinstance(e, ModelArtifactsNotAvailableError):
                    self.production_artifact_unavailable = True
                    self.reset_prod_model_version = e.model_version
                logging.error(f"Production model for {self.experiment_name} could not be found."
                              f"The service will set current model to Production.")
