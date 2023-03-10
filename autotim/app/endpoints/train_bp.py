import os
from flask import Blueprint, request
from injector import inject
from flask_api import status

from autotim.app.endpoints.utils.model_creation_utils import train
from autotim.storage_client.file_store_manager import FileStoreManager


TRAIN_BP = Blueprint('train', __name__)

@inject
@TRAIN_BP.route('/train', methods=['GET'])
def training(file_client: FileStoreManager):
    """Start training."""
    if "use_case_name" not in request.args \
            or "dataset_identifier" not in request.args:
        return "Request does not contain a use case name and/or dataset identifier.", \
               status.HTTP_404_NOT_FOUND

    train_size = "0.6"
    if "train_size" in request.args:
        if float(request.args["train_size"]) < 0 or float(request.args["train_size"]) > 1:
            return "train_size parameter must be float between 0.0 and 1.0", \
                   status.HTTP_400_BAD_REQUEST
        train_size = request.args["train_size"]

    name = request.args["use_case_name"]
    identifier = request.args["dataset_identifier"]

    evaluation_identifier = request.args.get("evaluation_identifier", None)

    column_label = request.args.get("column_label", "label")
    os.environ["COLUMN_LABEL"] = column_label

    column_id = request.args.get("column_id", "id")
    os.environ["COLUMN_ID"] = column_id

    column_sort = request.args.get("column_sort", "time")
    os.environ["COLUMN_SORT"] = column_sort

    column_value = request.args.get("column_value", "")
    os.environ["COLUMN_VALUE"] = column_value

    column_kind = request.args.get("column_kind", "")
    os.environ["COLUMN_KIND"] = column_kind

    recall_average = request.args.get("recall_average", "micro")
    os.environ["RECALL_AVERAGE"] = recall_average

    metric = request.args.get("metric", "accuracy")
    os.environ["METRIC"] = metric

    max_features = request.args.get("max_features", "1000")
    os.environ["MAX_FEATURES"] = max_features

    features_decrement = request.args.get("features_decrement", "0.9")
    os.environ["FEATURES_DECREMENT"] = features_decrement

    train_time = request.args.get("train_time", "dynamic")
    os.environ["TRAIN_TIME"] = train_time

    max_attempts = request.args.get("max_attempts", "5")

    return train(name=name, identifier=identifier, train_size=float(train_size),
                 file_client=file_client, max_attempts=max_attempts,
                 evaluation_identifier=evaluation_identifier)
