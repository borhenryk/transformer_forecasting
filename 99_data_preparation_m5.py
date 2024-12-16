# Databricks notebook source
# MAGIC %md
# MAGIC This notebook downloads the [M4 dataset](https://www.sciencedirect.com/science/article/pii/S0169207019301128) using an open source package: [`datasetsforecast`](https://github.com/Nixtla/datasetsforecast/tree/main/). The M4 dataset is a large and diverse collection of time series data used for benchmarking and evaluating the performance of forecasting methods. It is part of the [M-competition](https://forecasters.org/resources/time-series-data/) series, which are organized competitions aimed at comparing the accuracy and robustness of different forecasting methods.
# MAGIC
# MAGIC This notebook is run by other notebooks using `%run` command.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install package

# COMMAND ----------

# MAGIC %pip install datasetsforecast==0.0.8 --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Set the logging level

# COMMAND ----------

import pathlib
import pandas as pd
from datasetsforecast.m5 import M5
import logging

logger = spark._jvm.org.apache.log4j

# Setting the logging level to ERROR for the "py4j.java_gateway" logger
# This reduces the verbosity of the logs by only showing error messages
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)
logging.getLogger("py4j.clientserver").setLevel(logging.ERROR)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a catalog and a database
# MAGIC We create a catalog and a database (schema) to store the delta tables for our data.

# COMMAND ----------

default_catalog = "natyra_ts"
default_schema = "natyra_ts_fm"
default_samples = "0"

# Creating a text widget for the catalog name input
dbutils.widgets.text("catalog", default_catalog)
# Creating a text widget for the database (schema) name input
dbutils.widgets.text("db", default_schema)
# Creating a text widget for the number of time series to sample input
dbutils.widgets.text("n", default_samples)  # Default value set to "0"

catalog = dbutils.widgets.get("catalog") or default_catalog  # Name of the catalog we use to manage our assets
db = dbutils.widgets.get("db") or default_schema # Name of the schema we use to store assets
n = int(dbutils.widgets.get("n") or default_samples)  # Number of time series to sample, default to 0 if empty


# Ensure the catalog exists, create it if it does not
_ = spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
# Ensure the schema exists within the specified catalog, create it if it does not
_ = spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{db}")


# COMMAND ----------

# MAGIC %md
# MAGIC ##M5

# COMMAND ----------

# Load the M5 dataset
calendar_df, sales_df, prices_df = M5.load(directory=str(pathlib.Path.home()))


# COMMAND ----------

sales_sdf = spark.createDataFrame(sales_df)
prices_sdf = spark.createDataFrame(prices_df)
calendar_sdf = spark.createDataFrame(calendar_df)



# COMMAND ----------

sales_sdf.printSchema()
prices_sdf.printSchema()
calendar_sdf.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Calendar DF Transformation

# COMMAND ----------

from pyspark.sql.functions import to_date

calendar_sdf = calendar_sdf.withColumn("date", to_date("ds"))


# COMMAND ----------

display(calendar_sdf.head(100))

# COMMAND ----------

from pyspark.sql.functions import col, when, count

def check_nulls(sdf):
    null_counts = (
        sdf.select(
            [
                count(when(col(c).isNull(), c)).alias(c)  # Count nulls for each column
                for c in sdf.columns
            ]
        )
    )
    null_counts.show()



# COMMAND ----------

check_nulls(sales_sdf)
check_nulls(calendar_sdf)
check_nulls(prices_sdf)


# COMMAND ----------

calendar_sdf.write.format("delta").mode("overwrite").saveAsTable("natyra_ts.natyra_ts_fm.calendar")
prices_sdf.write.format("delta").mode("overwrite").saveAsTable("natyra_ts.natyra_ts_fm.prices")


# COMMAND ----------

# MAGIC %md
# MAGIC ##Sales DF transformation

# COMMAND ----------

from pyspark.sql.functions import col, when

# Define the columns
snap_cols = ['snap_CA', 'snap_TX', 'snap_WI']
event_cols = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']

# Cast snap columns to string
sales_sdf = sales_sdf.withColumn("snap_CA", col("snap_CA").cast("string")) \
                   .withColumn("snap_TX", col("snap_TX").cast("string")) \
                   .withColumn("snap_WI", col("snap_WI").cast("string"))

# Melt the DataFrame to long format
sales_sdf_long = sales_sdf.selectExpr(
    "unique_id", "ds", "sell_price",
    "stack(7, 'event_name_1', event_name_1, 'event_type_1', event_type_1, 'event_name_2', event_name_2, 'event_type_2', event_type_2, 'snap_CA', snap_CA, 'snap_TX', snap_TX, 'snap_WI', snap_WI) as (variable, value)"
)

# Create separate columns for event_name, event_type, and snap
sales_sdf_long = sales_sdf_long.withColumn(
    "event_name", when(col("variable").like("event_name%"), col("value"))
).withColumn(
    "event_type", when(col("variable").like("event_type%"), col("value"))
).withColumn(
    "snap", when(col("variable").like("snap%"), col("value"))
)

# Drop unnecessary columns
sales_sdf_long = sales_sdf_long.drop("variable", "value")

# Display the result
display(sales_sdf_long.limit(1000))

# COMMAND ----------

check_nulls(sales_sdf_long)

# COMMAND ----------

sales_long = sales_sdf_long.filter(~(col("event_name").isNull() & col("event_type").isNull() & col("snap").isNull()))

# COMMAND ----------

sales_long.write.format("delta").mode("overwrite").saveAsTable("natyra_ts.natyra_ts_fm.sales")


# COMMAND ----------

# MAGIC %md 
# MAGIC ##Merge Sales, Calendar, Prices in one dataframe

# COMMAND ----------

merged_sdf = sales_long.join(
    prices_sdf.select('unique_id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'),
    on='unique_id',
    how='left'
)

# Perform the second join: merged_sdf with calendar_df on 'unique_id' and 'ds'
final_sdf = merged_sdf.join(
    calendar_sdf.select('unique_id', 'ds', 'y', 'date'),
    on=['unique_id', 'ds'],
    how='left'
)

# COMMAND ----------

display(final_sdf.head(100))

# COMMAND ----------

final_sdf.write.format("delta").mode("overwrite").saveAsTable("natyra_ts.natyra_ts_fm.m5_final_merged")

# COMMAND ----------

import pyspark.pandas as ps

final_df = ps.read_table('natyra_ts.natyra_ts_fm.m5_final_merged')

# COMMAND ----------

# MAGIC %pip install jax[cuda12]==0.4.26 --quiet
# MAGIC %pip install protobuf==3.20.* --quiet
# MAGIC %pip install utilsforecast --quiet
# MAGIC %pip install numba --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

%pip install timesfm[pax]

# COMMAND ----------


import timesfm

# For PAX
tfm = timesfm.TimesFm(
      hparams=timesfm.TimesFmHparams(
          backend="gpu",
          per_core_batch_size=32,
          horizon_len=128,
      ),
      checkpoint=timesfm.TimesFmCheckpoint(
          huggingface_repo_id="google/timesfm-1.0-200m"),
  )

display(tfm)

# COMMAND ----------

inputs = ds.rename(columns = {'start_station_name':'unique_id', 'starttime':'ds'}),
    freq = "D",
    value_name = 'num_trips'

# COMMAND ----------

# Generate forecasts on the input DataFrame.
forecast_df = tfm.forecast_on_df(
    inputs=final_df,  # The input DataFrame containing the time series data.
    freq="D",  # Frequency of the time series data, set to daily.
    value_name="y",  # Column name in the DataFrame containing the values to forecast.
    num_jobs=-1,  # Number of parallel jobs to run, set to -1 to use all available processors.
)

# Display the forecast DataFrame.
display(forecast_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Daily M4 data
# MAGIC Below are some custom functions to convert the downloaded M4 time series into a daily format. The parameter `n` specifies the number of time series to sample for your dataset.

# COMMAND ----------

def create_m4_daily():
    # Load the M4 daily dataset
    y_df, _, _ = M4.load(directory=str(pathlib.Path.home()), group="Daily")
    # Create a list of unique IDs for the time series we want to sample
    _ids = [f"D{i}" for i in range(1, n)]
    # Filter and transform the dataset based on the unique IDs
    y_df = (
        y_df.groupby("unique_id")
        .filter(lambda x: x.unique_id.iloc[0] in _ids)
        .groupby("unique_id")
        .apply(transform_group_daily)
        .reset_index(drop=True)
    )
    return y_df


def transform_group_daily(df):
    unique_id = df.unique_id.iloc[0]  # Get the unique ID of the current group
    if len(df) > 1020:
        df = df.iloc[-1020:]  # Limit the data to the last 1020 entries if longer
    _start = pd.Timestamp("2020-01-01")  # Start date for the transformed data
    _end = _start + pd.DateOffset(days=int(df.count()[0]) - 1)  # End date for the transformed data
    date_idx = pd.date_range(start=_start, end=_end, freq="D", name="ds")  # Generate the date range
    res_df = pd.DataFrame(data=[], index=date_idx).reset_index()  # Create an empty DataFrame with the date range
    res_df["unique_id"] = unique_id  # Add the unique ID column
    res_df["y"] = df.y.values  # Add the target variable column
    return res_df

 
(
    spark.createDataFrame(create_m4_daily())  # Create a Spark DataFrame from the transformed data
    .write.format("delta").mode("overwrite")  # Write the DataFrame to Delta format, overwriting any existing data
    .saveAsTable(f"{catalog}.{db}.m4_daily_train")  # Save the table in the specified catalog and schema
)

# Print a confirmation message
print(f"Saved data to {catalog}.{db}.m4_daily_train")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Monthly M4 data
# MAGIC In our example notebooks, we primarily use daily time series. However, if you want to experiment with monthly time series, use the `m4_monthly_train` table generated by the following command.

# COMMAND ----------

def create_m4_monthly():
    # Load the M4 monthly dataset
    y_df, _, _ = M4.load(directory=str(pathlib.Path.home()), group="Monthly")
    # Create a list of unique IDs for the time series we want to sample
    _ids = [f"M{i}" for i in range(1, n + 1)]
    # Filter and transform the dataset based on the unique IDs
    y_df = (
        y_df.groupby("unique_id")
        .filter(lambda x: x.unique_id.iloc[0] in _ids)
        .groupby("unique_id")
        .apply(transform_group_monthly)
        .reset_index(drop=True)
    )
    return y_df


def transform_group_monthly(df):
    unique_id = df.unique_id.iloc[0]  # Get the unique ID of the current group
    _cnt = 60  # Set the count for the number of months
    _start = pd.Timestamp("2018-01-01")  # Start date for the transformed data
    _end = _start + pd.DateOffset(months=_cnt)  # End date for the transformed data
    date_idx = pd.date_range(start=_start, end=_end, freq="M", name="date")  # Generate the date range for monthly data
    _df = (
        pd.DataFrame(data=[], index=date_idx)  # Create an empty DataFrame with the date range
        .reset_index()
        .rename(columns={"index": "date"})  # Rename the index column to "date"
    )
    _df["unique_id"] = unique_id  # Add the unique ID column
    _df["y"] = df[:60].y.values  # Add the target variable column, limited to 60 entries
    return _df


(
    spark.createDataFrame(create_m4_monthly())  # Create a Spark DataFrame from the transformed data
    .write.format("delta").mode("overwrite")  # Write the DataFrame to Delta format, overwriting any existing data
    .saveAsTable(f"{catalog}.{db}.m4_monthly_train")  # Save the table in the specified catalog and schema
)

# Print a confirmation message
print(f"Saved data to {catalog}.{db}.m4_monthly_train")


# COMMAND ----------

# MAGIC %md
# MAGIC © 2024 Databricks, Inc. All rights reserved. 
# MAGIC
# MAGIC The sources in all notebooks in this directory and the sub-directories are provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | datasetsforecast | Datasets for Time series forecasting | MIT | https://pypi.org/project/datasetsforecast/
