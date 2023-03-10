"""Represent AutoTiM models. """


class AutoTiM_Model:
    """Represent AutoTiM models. """
    mlflow_uri: str
    autotim_model_name: str
    model_version: int
    model = None
    feature_settings = None
    params = None
    run_id = None
    stage = None

    def __init__(self, use_case_name: str, dataset_identifier: str,
                 model_version: int = None,
                 mlflow_model = None,
                 feature_settings = None,
                 model_params = None,
                 run_id = None,
                 stage: str = None):
        self.autotim_model_name = \
            AutoTiM_Model.get_autotim_convention_model_name(
                use_case_name=use_case_name, dataset_identifier=dataset_identifier)

        self.mlflow_uri = AutoTiM_Model.get_mlflow_model_uri(model_name=self.autotim_model_name,
                                                            model_version=model_version,
                                                            stage=stage)

        self.model_version = model_version
        self.model = mlflow_model
        self.feature_settings = feature_settings
        self.params = model_params
        self.run_id = run_id

        if self.model_version is None:
            self.stage = stage

    @staticmethod
    def get_autotim_convention_model_name(use_case_name: str, dataset_identifier: str):
        return f"{use_case_name}-{dataset_identifier}_model"

    @staticmethod
    def get_mlflow_model_uri_with_version(model_name: str, model_version: int):
        return f"models:/{model_name}/{model_version}"

    @staticmethod
    def get_mlflow_model_uri_by_stage(model_name: str, stage):
        return f'models:/{model_name}/{stage}'

    @staticmethod
    def get_mlflow_model_uri(model_name: str, model_version: str = None, stage: str = 'Production'):
        return AutoTiM_Model.get_mlflow_model_uri_with_version(
            model_name=model_name, model_version=model_version) \
            if model_version is not None \
            else AutoTiM_Model.get_mlflow_model_uri_by_stage(model_name=model_name, stage=stage)
