# Databricks notebook source
# MAGIC %md
# MAGIC # Installation of packages

# COMMAND ----------

# MAGIC %pip install timesfm[torch] --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Data preparation

# COMMAND ----------

catalog = "natyra_ts"  # Name of the catalog we use to manage our assets
db = "natyrs_ts_fm"  # Name of the schema we use to manage our assets (e.g. datasets)
n = 100  # Number of time series to sample

# COMMAND ----------

# This cell runs the notebook ../data_preparation and creates the following tables with M4 data: 
# 1. {catalog}.{db}.m4_daily_train
# 2. {catalog}.{db}.m4_monthly_train
dbutils.notebook.run("./transformer_forecasting_9", timeout_seconds=0, arguments={"catalog": catalog, "db": db, "n": n})

# COMMAND ----------

df = spark.table(f'{catalog}.{db}.m4_daily_train').toPandas()
display(df)

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # TimesFM

# COMMAND ----------

# MAGIC %pip install numba
# MAGIC

# COMMAND ----------

# Install the required package

# Initialize the TimesFm model with specified parameters.
import timesfm
tfm = timesfm.TimesFm(
      hparams=timesfm.TimesFmHparams(
          backend="gpu",
          per_core_batch_size=32,
          horizon_len=128,
      ),
      checkpoint=timesfm.TimesFmCheckpoint(
          huggingface_repo_id="google/timesfm-1.0-200m-pytorch"),
  )

# COMMAND ----------

# Generate forecasts on the input DataFrame.
forecast_df = tfm.forecast_on_df(
    inputs=df,  # The input DataFrame containing the time series data.
    freq="D",  # Frequency of the time series data, set to daily.
    value_name="y",  # Column name in the DataFrame containing the values to forecast.
    num_jobs=-1,  # Number of parallel jobs to run, set to -1 to use all available processors.
)

# Display the forecast DataFrame.
display(forecast_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Registration

# COMMAND ----------

import mlflow
import torch
import numpy as np
from mlflow.models import infer_signature
from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, TensorSpec
from pyspark.sql.functions import current_user


# Set the MLflow registry URI to Databricks Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# Get the current user name and store it in a variable
current_user_name = spark.sql("SELECT current_user()").collect()[0][0]

# Set the experiment name
mlflow.set_experiment(f"/Users/{current_user_name}/tsfm")

# Define a custom MLflow Python model class for TimesFM
class TimesFMModel(mlflow.pyfunc.PythonModel):
    def __init__(self, repository):
        self.repository = repository  # Store the repository ID for the model checkpoint
        self.tfm = None  # Initialize the model attribute to None

    def load_model(self):
        import timesfm
        # Initialize the TimesFm model with specified parameters
        self.tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="gpu",
                per_core_batch_size=32,
                horizon_len=128,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id=self.repository
                ),
            )

    def predict(self, context, input_df, params=None):
        # Load the model if it hasn't been loaded yet
        if self.tfm is None:
            self.load_model()
        # Generate forecasts on the input DataFrame
        forecast_df = self.tfm.forecast_on_df(
            inputs=input_df,  # Input DataFrame containing the time series data.
            freq="D",  # Frequency of the time series data, set to daily.
            value_name="y",  # Column name in the DataFrame containing the values to forecast.
            num_jobs=-1,  # Number of parallel jobs to run, set to -1 to use all available processors.
        )
        return forecast_df  # Return the forecast DataFrame

    def __getstate__(self):
        state = self.__dict__.copy()  # Copy the instance's state
        # Remove the tfm attribute from the state, as it's not serializable
        del state['tfm']
        return state

    def __setstate__(self, state):
        # Restore instance attributes
        self.__dict__.update(state)
        # Reload the model since it was not stored in the state
        self.load_model()

# COMMAND ----------

# Initialize the custom TimesFM model with the specified repository ID
pipeline = TimesFMModel("google/timesfm-1.0-200m-pytorch")
# Infer the model signature based on input and output DataFrames
signature = infer_signature(
    model_input=df,  # Input DataFrame for the model
    model_output=pipeline.predict(None, df),  # Output DataFrame from the model
)

# Define the registered model name using variables for catalog and database
registered_model_name = f"{catalog}.{db}.timesfm-1-200m-pytorch"

# Start an MLflow run to log and register the model
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        "model",  # The artifact path where the model is logged
        python_model=pipeline,  # The custom Python model to log
        registered_model_name=registered_model_name,  # The name to register the model under
        signature=signature,  # The model signature
        input_example=df,  # An example input to log with the model
        pip_requirements=["timesfm[torch]"],
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load model

# COMMAND ----------

from mlflow import MlflowClient
client = MlflowClient()

# Define a function to get the latest version number of a registered model
def get_latest_model_version(client, registered_model_name):
    latest_version = 1  # Initialize the latest version number to 1
    # Iterate through all model versions of the specified registered model
    for mv in client.search_model_versions(f"name='{registered_model_name}'"):
        version_int = int(mv.version)  # Convert the version number to an integer
        if version_int > latest_version:  # Check if the current version is greater than the latest version
            latest_version = version_int  # Update the latest version number
    return latest_version  # Return the latest version number

# Get the latest version number of the specified registered model
model_version = get_latest_model_version(client, registered_model_name)
# Construct the model URI using the registered model name and the latest version number
logged_model = f"models:/{registered_model_name}/{model_version}"

# Load the model as a PyFuncModel
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Generate forecasts using the loaded model on the input DataFrame
loaded_model.predict(df)  # Use the loaded model to make predictions on the input DataFrame

# COMMAND ----------

# MAGIC %md
# MAGIC #### Deploy model

# COMMAND ----------

# With the token, you can create our authorization header for our subsequent REST calls
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# Next you need an endpoint at which to execute your request which you can get from the notebook's tags collection
java_tags = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags()

# This object comes from the Java CM - Convert the Java Map opject to a Python dictionary
tags = sc._jvm.scala.collection.JavaConversions.mapAsJavaMap(java_tags)

# Lastly, extract the Databricks instance (domain name) from the dictionary
instance = tags["browserHostName"]

# COMMAND ----------

import requests

model_serving_endpoint_name = "timesfm-1-200m-pytorch"

my_json = {
    "name": model_serving_endpoint_name,
    "config": {
        "served_models": [
            {
                "model_name": registered_model_name,
                "model_version": model_version,
                "workload_type": "GPU_SMALL",
                "workload_size": "Small",
                "scale_to_zero_enabled": "true",
                "environment_vars":{
                    "JAX_PLATFORMS": "gpu"
                }
            }
        ],
        "auto_capture_config": {
            "catalog_name": catalog,
            "schema_name": db,
            "table_name_prefix": model_serving_endpoint_name,
        },
    },
}

# Make sure to drop the inference table of it exists
_ = spark.sql(
    f"DROP TABLE IF EXISTS {catalog}.{db}.`{model_serving_endpoint_name}_payload`"
)

# COMMAND ----------

# Function to create an endpoint in Model Serving and deploy the model behind it
def func_create_endpoint(model_serving_endpoint_name):
    # get endpoint status
    endpoint_url = f"https://{instance}/api/2.0/serving-endpoints"
    url = f"{endpoint_url}/{model_serving_endpoint_name}"
    r = requests.get(url, headers=headers)
    if "RESOURCE_DOES_NOT_EXIST" in r.text:
        print(
            "Creating this new endpoint: ",
            f"https://{instance}/serving-endpoints/{model_serving_endpoint_name}/invocations",
        )
        re = requests.post(endpoint_url, headers=headers, json=my_json)
    else:
        new_model_version = (my_json["config"])["served_models"][0]["model_version"]
        print(
            "This endpoint existed previously! We are updating it to a new config with new model version: ",
            new_model_version,
        )
        # update config
        url = f"{endpoint_url}/{model_serving_endpoint_name}/config"
        re = requests.put(url, headers=headers, json=my_json["config"])
        # wait till new config file in place
        import time, json

        # get endpoint status
        url = f"https://{instance}/api/2.0/serving-endpoints/{model_serving_endpoint_name}"
        retry = True
        total_wait = 0
        while retry:
            r = requests.get(url, headers=headers)
            assert (
                r.status_code == 200
            ), f"Expected an HTTP 200 response when accessing endpoint info, received {r.status_code}"
            endpoint = json.loads(r.text)
            if "pending_config" in endpoint.keys():
                seconds = 10
                print("New config still pending")
                if total_wait < 6000:
                    # if less the 10 mins waiting, keep waiting
                    print(f"Wait for {seconds} seconds")
                    print(f"Total waiting time so far: {total_wait} seconds")
                    time.sleep(10)
                    total_wait += seconds
                else:
                    print(f"Stopping,  waited for {total_wait} seconds")
                    retry = False
            else:
                print("New config in place now!")
                retry = False

    assert (
        re.status_code == 200
    ), f"Expected an HTTP 200 response, received {re.status_code}"

# Function to delete the endpoint from Model Serving
def func_delete_model_serving_endpoint(model_serving_endpoint_name):
    endpoint_url = f"https://{instance}/api/2.0/serving-endpoints"
    url = f"{endpoint_url}/{model_serving_endpoint_name}"
    response = requests.delete(url, headers=headers)
    if response.status_code != 200:
        raise Exception(
            f"Request failed with status {response.status_code}, {response.text}"
        )
    else:
        print(model_serving_endpoint_name, "endpoint is deleted!")
    return response.json()

# COMMAND ----------

# Create an endpoint. This may take some time.
func_create_endpoint(model_serving_endpoint_name)

# COMMAND ----------

import time, mlflow

# Define a function to wait for a serving endpoint to be ready
def wait_for_endpoint():
    endpoint_url = f"https://{instance}/api/2.0/serving-endpoints"  # Construct the base URL for the serving endpoints API
    while True:  # Infinite loop to repeatedly check the status of the endpoint
        url = f"{endpoint_url}/{model_serving_endpoint_name}"  # Construct the URL for the specific model serving endpoint
        response = requests.get(url, headers=headers)  # Send a GET request to the endpoint URL with the necessary headers
        
        # Ensure the response status code is 200 (OK)
        assert (
            response.status_code == 200
        ), f"Expected an HTTP 200 response, received {response.status_code}\n{response.text}"

        # Extract the status of the endpoint from the response JSON
        status = response.json().get("state", {}).get("ready", {})
        # print("status",status)  # Optional: Print the status for debugging purposes
        
        # Check if the endpoint status is "READY"
        if status == "READY":
            print(status)  # Print the status if the endpoint is ready
            print("-" * 80)  # Print a separator line for clarity
            return  # Exit the function when the endpoint is ready
        else:
            # Print a message indicating the endpoint is not ready and wait for 5 minutes
            print(f"Endpoint not ready ({status}), waiting 5 minutes")
            time.sleep(300)  # Wait for 300 seconds before checking again

# Get the Databricks web application URL using MLflow utility function
api_url = mlflow.utils.databricks_utils.get_webapp_url()

# Call the wait_for_endpoint function to wait for the serving endpoint to be ready
wait_for_endpoint()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Query the online forecast

# COMMAND ----------

import os
import requests
import pandas as pd
import json
import matplotlib.pyplot as plt

# Replace URL with the end point invocation url you get from Model Seriving page.
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
def create_tf_serving_json(data):
    return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
    url = f'https://{instance}/serving-endpoints/{model_serving_endpoint_name}/invocations'
    headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
    ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.request(method='POST', headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    return response.json()

# COMMAND ----------

display(df)

# COMMAND ----------

df['ds'] = df['ds'].astype(str)

# COMMAND ----------

score_model(df)
