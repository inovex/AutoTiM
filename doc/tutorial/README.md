# Tutorial
This tutorial uses the [robot-failure-dataset][1] to walk you through the entire functionality of our AutoTiM service step-by-step. The tutorial includes:
- Preparing your dataset to match our requirements
- Storing a dataset using the `/store`-endpoint
- Training a model using the `/train`-endpoint
- Performing a prediction with the trained model using the `/predict`-endpoint

**Prerequisities:**
- Please setup your local service as described [here](../../README.md) and keep your docker containers running for this tutorial.
- In your terminal, change the directory into `./tutorial`.
<br><br><br>
***

### Step 1 - Prepare your dataset
To use our service, your dataset must contain:
    - a column with id's, which assigns each row to a timeseries (default column name: `id`)
    - a column with timestamps or sortable digits, which orders the datapoints of a timeseries timewise (default column name: `time`)
    - a column with labels, which assigns the label of the timeseries to each datapoint (default column name: `label`)

Let's have a look at the roboter dataset (stored in *./dataset/data.csv*) to further explain these requirements:
| |id |time|F_x|F_y|F_z|T_x|T_y|T_z|label|
|------|---|----|---|---|---|---|---|---|-----|
|0     |1  |0   |-1 |-1 |63 |-3 |-1 |0  |True |
|1     |1  |1   |0  |0  |62 |-3 |-1 |0  |True |
|2     |1  |2   |-1 |-1 |61 |-3 |0  |0  |True |
|3     |1  |3   |-1 |-1 |63 |-2 |-1 |0  |True |
|4     |1  |4   |-1 |-1 |63 |-3 |-1 |0  |True |
|...   |...|... |...|...|...|...|...|...|...  |

The dataset contains timeseries of robots whose behaviors were recorded by six sensors. The corresponding features are *F_x, F_y, F_z, T_x, T_y, T_z*.<br><br>
There is one timeseries for each robot, uniquely identified by the column `id`. The timeseries themselves are sorted by the column `time`. The column `label` contains the information whether the robot failed or not.<br><br>
For this tutorial, the dataset already contains all required columns and are given the default name. When uploading your own dataset, you need to make sure that it meets all above requirements and provide the names of the columns in your requests if they differ from the default name.

<br><br><br>
***

### Step 2 - Store the dataset using the /store-endpoint
The dataset is stored in *./dataset/data.csv*. To train a model on the dataset, you first need make it available to our service using our /store-endpoint. This will store it in a docker volume of your running AutoTiM docker container. To do so, run the following command in your terminal:

```shell
curl -i -X POST --user admin:password -F "file=@./dataset/data.csv" -F "use_case_name=roboter_failures" -F "dataset_identifier=tutorial" http://localhost:5004/store
```

Once the storing is complete, you will receive the following feedback:
```
{
    "use_case_name": "roboter_failures",
    "dataset_identifier": "tutorial"
}
```
<br><br><br>
***
Note that the dataset is stored permanently and does not need to be uploaded again before each training. If you wish to upload different versions of your dataset for a particular use case, you can do so by uploading another dataset with the same use case name and a different dataset identifier.

### Step 3 - Train a model on your dataset using the /train-endpoint
You can now train a model on the robot dataset. To do so, call the following URL in your browser:

```shell
http://127.0.0.1:5004/train?use_case_name=roboter_failures&dataset_identifier=tutorial
```

<details>
  <summary><em>Customize your training request to match your use case requirements</em></summary><blockquote>
You can customize your training using the optional parameters. For detailed documentation go [here](../README.md#222-train).<br>
The following parameters can be set:

##### Optional parameters
`column_id`: Name of the id column, which assigns each row to a timeseries (default column name: id)<br>
`column_label`: Name of the column containing the classification labels (default: label)<br>
`column_sort`: Name of the column that contains values which allow to sort the time series, e.g. time stamps (default: time)<br>
`column_value`: Name of the column that contains the actual values of the time series, e.g. sensor data (default: None)<br>
`column_kind`: Name of the column that indicates the names of the different timeseries types, e.g. different sensors (default: None)<br>
`train_size`: Proportion of the dataset to include in the train data when performing the train-test-split (default: 0.6)<br>
`recall_average`: Metric to be used to calculate the recall and precision score (default: micro; possible metrics are: micro, macro, samples, weighted, binary or None)<br>
`metric`: Metric to be used for the model selection (default: accuracy; possible metrics are: accuracy, balanced_accuracy, recall_score, precision_score) <br>
`max_features`: Maximum number of features used for training (default: 1000)<br>
`features_decrement`: Decrement step of features when a recursion error occurs. <br> If smaller then `1` this will be percentage based otherwise it will be an absolute value (default: 0.9) <br>
`max_attempts`: Maximum number of attempts for training when failing due to a recursion error. (default: 5)<br>
`train_time`: Time in minutes used for training the model. If not specified it uses the dynamic training time. (default: dynamic)<br>
`evaluation_identifier`: Name of the dataset within your project, only used for evaluation. If specified, `dataset_identifier` only used for training.

##### Example use
```shell
http://127.0.0.1:5004/train?use_case_name=roboter_failures&dataset_identifier=tutorial&column_label=classes
```

</blockquote></details>

You may need to enter a username and password. Use *admin* and *password*. <br>
The training may take a while. Once it is finished, you will get feedback that the training has been completed including training details such as the latest trained model version, production model version, evaluation metrics for one or both, warnings etc. in your browser interface.

E.g. for the first trained model in the tutorial the response can look like this:
```
{
    "training": "completed",
    "latest_model_version": "1"
}
```
You can now view the trained model on MLFlow at http://0.0.0.0:5000 (or http://localhost:5000 for Windows). It is logged under both *Experiments* and *Models* under the name *"roboter_failures-tutorial"*. If you click on the model, you can check its metadata and performance on the dataset (accuracy, balanced accuracy, number of extracted features, precision and recall score). You can also download the model directly - you find it under *Artifacts* in the model view.
<br><br><br>

***

### Step 4 - Predict with your model using the /predict-endpoint
To predict the classifications of timeseries with your model, we will use the timeseries stored in *./dataset/predict.csv* which contains two timeseries. The predict dataset is structured identically to the training dataset, except it is missing the label-column. To perform the prediction, run the following command in your terminal:
```shell
curl -i -X POST --user admin:password -F "file=@./dataset/predict.csv" -F "use_case_name=roboter_failures" -F "dataset_identifier=tutorial" http://localhost:5004/predict
```
Your response will look like this, showing you the predicted label for both timeseries:
```
{
    "prediction": [
        true,
        false
    ]
}
```
And that's it! You trained your first model with our AutoTiM service 👏🏽
<br><br><br><br><br><br>



[1]: <http://archive.ics.uci.edu/ml/datasets/Robot+Execution+Failures>
