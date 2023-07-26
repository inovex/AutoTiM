import os
import pandas as pd

from autotim.feature_engineering.automated_feature_engineering import create_features
from autotim.app.endpoints.utils.model_creation_utils import train
from autotim.prediction_service.autotim_prediction_service import AutoTiMPredictionService


# read file
data_folder = "data"
path = f"{data_folder}/data.csv"
timeseries = pd.read_csv(f"{data_folder}/predict.csv")

# set params
name="test"
identifier="test"
os.environ["COLUMN_LABEL"] = "label"
os.environ["COLUMN_ID"] = "id"
os.environ["COLUMN_SORT"] = "time"
# os.environ["COLUMN_VALUE"] = None
# os.environ["COLUMN_KIND"] = None

# train params
os.environ["RECALL_AVERAGE"] = "micro"
os.environ["METRIC"] = "accuracy"
os.environ["MAX_FEATURES"] = "1000"
os.environ["FEATURES_DECREMENT"] = "0.9"
os.environ["TRAIN_TIME"] = "dynamic"
max_attempts = "1" # TODO
train_size = 0.6
evaluation_identifier = None

# training
result = train(name=name, identifier=identifier, path=path, train_size=float(train_size),
             max_attempts=max_attempts,
             evaluation_identifier=evaluation_identifier)

# prediction
autotim_prediction_service = AutoTiMPredictionService()

use_case_name = name
dataset_identifier = identifier
model_version = 1# request.form.get('model_version', None) #TODO

autotim_model = autotim_prediction_service.get_model(use_case_name=use_case_name,
                                                   dataset_identifier=dataset_identifier,
                                                   model_version=model_version)

features = create_features(dataframe=timeseries, settings=autotim_model.feature_settings,
                           column_id=os.getenv("COLUMN_ID"),
                           column_value=os.getenv("COLUMN_VALUE"),
                           column_kind=os.getenv("COLUMN_KIND"),
                           column_sort=os.getenv("COLUMN_SORT"))

prediction = autotim_prediction_service.predict(features=features,
                                               model=autotim_model.model)

print(prediction)
print("DONE")
