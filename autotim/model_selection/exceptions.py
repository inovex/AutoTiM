"""Exceptions for Model Selection. """


class MlflowExperimentNotFoundError(Exception):
    """Raised when trying to access an MlFlow model that does not exist."""

    def __init__(self, experiment_name: str):
        self.model_name = experiment_name
        self.message = f"The experiment '{experiment_name}' was not found."
        super().__init__(self.message)


class StageDoesNotExistError(Exception):
    """Raised when trying to access an MlFlow model that does not exist."""

    def __init__(self, stage_name: str):
        self.model_name = stage_name
        self.message = f"The experiment '{stage_name}' was not found. Please" \
                       f"choose either 'All', 'None', 'Staging' or 'Production'."
        super().__init__(self.message)


class VersionMustBeAnIntegerError(Exception):
    """Raised when trying to access an MlFlow model that does not exist."""

    def __init__(self, version):
        self.model_name = version
        self.message = f" '{version}' is not an integer. Please choose an integer" \
                       f"as the model version."
        super().__init__(self.message)


class MlflowModelNotFoundError(Exception):
    """Raised when trying to access a mlflow model that does not exist."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.message = f"The model '{model_name}' was not found."
        super().__init__(self.message)

class ModelSelectionFailed(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class ModelArtifactsNotAvailableError(Exception):
    def __init__(self, model_name, model_version):
        self.model_version = model_version
        model_description = model_name
        if self.model_version is not None:
            model_description = model_name + ", version" +str(model_version)

        self.message = f"Model {model_description} is logged in MlFlow, " \
                       f"but checkpoints and/or feature settings cannot be found " \
                       f"among the available artifacts."
        super().__init__(self.message)
