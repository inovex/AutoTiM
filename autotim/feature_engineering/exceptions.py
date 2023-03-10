"""Exceptions for Feature Engineering. """


class FeatureCreationFailedError(Exception):
    """Raised when encountering an exception when attempting to create features."""

    def __init__(self, message):
        self.message = "Could not create features from your input data: " + message
        super().__init__(self.message)

class DataSplitError(Exception):
    """Raised when encountering an exception when attempting to split the dataset."""

    def __init__(self, message, status):
        self.message = "Could not split the dataset from your input. " + message
        self.status = status
        super().__init__(self.message)
