# Databricks notebook source
# MAGIC %md # Data source
# MAGIC 
# MAGIC The common data elements for predictive maintenance problems can be summarized as follows:
# MAGIC 
# MAGIC * Machine features: The features specific to each individual machine, e.g. engine size, make, model, location, installation date.
# MAGIC * Telemetry data: The operating condition data collected from sensors, e.g. temperature, vibration, operating speeds, pressures.
# MAGIC * Maintenance history: The repair history of a machine, e.g. maintenance activities or component replacements, this can also include error code or runtime message logs.
# MAGIC * Failure history: The failure history of a machine or component of interest.
# MAGIC 
# MAGIC It is possible that failure history is contained within maintenance history, either as in the form of special error codes or order dates for spare parts. In those cases, failures can be extracted from the maintenance data. Additionally, different business domains may have a variety of other data sources that influence failure patterns which are not listed here exhaustively. These should be identified by consulting the domain experts when building predictive models.
# MAGIC 
# MAGIC Some examples of above data elements from use cases are:
# MAGIC     
# MAGIC **Machine conditions and usage:** Flight routes and times, sensor data collected from aircraft engines, sensor readings from ATM transactions, train events data, sensor readings from wind turbines, elevators and connected cars.
# MAGIC     
# MAGIC **Machine features:** Circuit breaker technical specifications such as voltage levels, geolocation or car features such as make, model, engine size, tire types, production facility etc.
# MAGIC 
# MAGIC **Failure history:** fight delay dates, aircraft component failure dates and types, ATM cash withdrawal transaction failures, train/elevator door failures, brake disk replacement order dates, wind turbine failure dates and circuit breaker command failures.
# MAGIC 
# MAGIC **Maintenance history:** Flight error logs, ATM transaction error logs, train maintenance records including maintenance type, short description etc. and circuit breaker maintenance records.
# MAGIC 
# MAGIC Given the above data sources, the two main data types we observe in predictive maintenance domain are temporal data and static data. Failure history, machine conditions, repair history, usage history are time series indicated by the timestamp of data collection. Machine and operator specific features, are more static, since they usually describe the technical specifications of machines or operator’s properties.
# MAGIC 
# MAGIC For this scenario, we use a relatively large-scale data to walk the user through the main steps from data ingestion (this Jupyter notebook), feature engineering, model building, and model operationalization and deployment. The code for the entire process is written in PySpark and implemented using Jupyter notebooks within Azure ML Workbench. The selected model is operationalized using Azure Machine Learning Model Management for use in a production environment simulating making realtime failure predictions. 
# MAGIC 
# MAGIC # Step 1: Data Ingestion
# MAGIC 
# MAGIC This data aquisiton notebook will download the simulated predicitive maintenance data sets from our GitHub data store. We do some preliminary data cleaning and verification, and store the results as a Spark data frame in an Azure Blob storage container for use in the remaining notebook steps of this analysis.
# MAGIC 
# MAGIC **Note:** This notebook will take about 10-15 minutes to execute all cells, depending on the compute configuration you have setup. Most of this time is spent handling the _telemetry_ data set, which contains about 8.7 million records.

# COMMAND ----------

## Setup our environment by importing required libraries
import time
import os
import glob
import urllib

# Read csv file from URL directly
import pandas as pd

# For creating some preliminary EDA plots.
# %matplotlib inline
import matplotlib.pyplot as plt
from ggplot import *

from datetime import datetime

# Setup the pyspark environment
from pyspark.sql import SparkSession

# Time the notebook execution. 
# This will only make sense if you "Run All" cells
tic = time.time()

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %md ## Download simulated data sets
# MAGIC We will be reusing the raw simulated data files from another tutorial. The notebook automatically downloads these files stored at [Microsoft/SQL-Server-R-Services-Samples GitHub site](https://github.com/Microsoft/SQL-Server-R-Services-Samples/tree/master/PredictiveMaintanenceModelingGuide/Data).
# MAGIC 
# MAGIC The five data files are:
# MAGIC 
# MAGIC  * machines.csv
# MAGIC  * maint.csv
# MAGIC  * errors.csv
# MAGIC  * telemetry.csv
# MAGIC  * failures.csv
# MAGIC 
# MAGIC To get an idea of what is contained in the data, we examine this machine schematic. 
# MAGIC ![Machine schematic](../images/machine.png)
# MAGIC 
# MAGIC There are 1000 machines of four different models. Each machine contains four components of interest, and four sensors measuring voltage, pressure, vibration and rotation. A controller monitors the system and raises alerts for five different error conditions. Maintenance logs indicate when something is done to the machine which does not include a component replacement. A failure is defined by the replacement of a component. 
# MAGIC 
# MAGIC This notebook does some preliminary data cleanup, creates summary graphics for each data set to verify the data downloaded correctly, and stores the resulting data sets in the Azure blob container created in the previous section.

# COMMAND ----------

# The raw data is stored on GitHub here:
basedataurl = "http://media.githubusercontent.com/media/Microsoft/SQL-Server-R-Services-Samples/master/PredictiveMaintanenceModelingGuide/Data/"

# We will store each of these data sets in blob storage in an 
# Azure Storage Container on your Azure subscription.
# See https://github.com/Azure/ViennaDocs/blob/master/Documentation/UsingBlobForStorage.md
# for details.

# These file names detail which blob each files is stored under. 
MACH_DATA = 'machines_files'
MAINT_DATA = 'maint_files'
ERROR_DATA = 'errors_files'
TELEMETRY_DATA = 'telemetry_files'
FAILURE_DATA = 'failure_files'


# COMMAND ----------

# MAGIC %md ### Machines data set
# MAGIC 
# MAGIC This simulation tracks a simulated set of 1000 machines over the course of a single year (2015). 
# MAGIC 
# MAGIC This data set includes information about each machine: Machine ID, model type and age (years in service). 

# COMMAND ----------

# load raw data from the GitHub URL
datafile = "machines.csv"

# Download the file once, and only once.
if not os.path.isfile(datafile):
    urllib.request.urlretrieve(basedataurl+datafile, datafile)
    
# Read into pandas
machines = pd.read_csv(datafile, encoding='utf-8')

print(machines.count())
machines.head(10)

# COMMAND ----------

# MAGIC %md The following figure plots a histogram of the machines age colored by the specific model.

# COMMAND ----------

plt.figure(figsize=(8, 6))

fig, ax = plt.subplots()

_, bins, _ = plt.hist([machines.loc[machines['model'] == 'model1', 'age'],
                       machines.loc[machines['model'] == 'model2', 'age'],
                       machines.loc[machines['model'] == 'model3', 'age'],
                       machines.loc[machines['model'] == 'model4', 'age']],
                       20, stacked=True, label=['model1', 'model2', 'model3', 'model4'])
plt.xlabel('Age (yrs)')
plt.ylabel('Count')
plt.legend()
display(fig)

# COMMAND ----------

# MAGIC %md The figure shows how long the collection of machines have been in service. It indicates there are four model types, shown in different colors, and all four models have been in service over the entire 20 years of service. The machine age will be a feature in our analysis, since we expect older machines may have a different set of errors and failures then machines that have not been in service long.
# MAGIC 
# MAGIC Next, we convert the machines data to a Spark dataframe, and verify the data types have converted correctly. 

# COMMAND ----------

# The data was read in using a Pandas data frame. We'll convert 
# it to pyspark to ensure it is in a Spark usable form for later 
# manipulations.
mach_spark = spark.createDataFrame(machines, 
                                   verifySchema=False)

# We no longer need th pandas dataframe, so we can release that memory.
del machines

# Check data type conversions.
mach_spark.printSchema()


# COMMAND ----------

# MAGIC %md Now we write the spark dataframe to an Azure blob storage container for use in the remaining notebooks of this scenario.

# COMMAND ----------

# Write the Machine data set to intermediate storage
mach_spark.write.mode('overwrite').saveAsTable(MACH_DATA)
#mach_spark.createOrReplaceTempView(MACH_DATA)
print("Machines files saved!")

# COMMAND ----------

# MAGIC %md ### Errors  data set
# MAGIC 
# MAGIC The error log contains non-breaking errors recorded while the machine is still operational. These errors are not considered failures, though they may be predictive of a future failure event. The error datetime field is rounded to the closest hour since the telemetry data (loaded later) is collected on an hourly rate.

# COMMAND ----------

# load raw data from the GitHub URL
datafile = "errors.csv"

# Download the file once, and only once.
if not os.path.isfile(datafile):
    urllib.request.urlretrieve(basedataurl+datafile, datafile)
    
# Read into pandas
errors = pd.read_csv(datafile, encoding='utf-8')

print(errors.count())
errors.head(10)

# COMMAND ----------

# MAGIC %md The following histogram details the distribution of the errors tracked in the log files. 

# COMMAND ----------

# Quick plot to show structure
pl = ggplot(aes(x="errorID"), errors) + geom_bar(fill="blue", color="black")
display(pl)

# COMMAND ----------

# MAGIC %md The error data consists of a time series (datetime stamped) of error codes thrown by each machine (machineID). The figure shows how many errors occured in each of the five error classes over the entire year. We could split this figure over each individual machine, but with 1000 individuals, the figure would not be very informative.
# MAGIC 
# MAGIC Next, we convert the errors data to a Spark dataframe, and verify the data types have converted correctly. 

# COMMAND ----------

# The data was read in using a Pandas data frame. We'll convert 
# it to pyspark to ensure it is in a Spark usable form for later 
# manipulations.
error_spark = spark.createDataFrame(errors, 
                               verifySchema=False)

# We no longer need the pandas dataframe, so we can release that memory.
del errors

# Check data type conversions.
error_spark.printSchema()

# COMMAND ----------

# MAGIC %md Now we write the spark dataframe to an Azure blob storage container for use in the remaining notebooks of this scenario.

# COMMAND ----------

# Write the Errors data set to intermediate storage
error_spark.write.mode('overwrite').saveAsTable(ERROR_DATA)

print("Errors files saved!")

# COMMAND ----------

# MAGIC %md ### Maintenance data set
# MAGIC 
# MAGIC The maintenance log contains both scheduled and unscheduled maintenance records. Scheduled maintenance corresponds with  regular inspection of components, unscheduled maintenance may arise from mechanical failure or other performance degradations. A failure record is generated for component replacement in the case  of either maintenance events. Because maintenance events can also be used to infer component life, the maintenance data has been collected over two years (2014, 2015) instead of only over the year of interest (2015).

# COMMAND ----------

# load raw data from the GitHub URL
datafile = "maint.csv"

# Download the file once, and only once.
if not os.path.isfile(datafile):
    urllib.request.urlretrieve(basedataurl+datafile, datafile)
    
# Read into pandas
maint = pd.read_csv(datafile, encoding='utf-8')

print(maint.count())
maint.head(20)

# COMMAND ----------

# Quick plot to show structure
pl = ggplot(aes(x="comp"), maint) + geom_bar(fill="blue", color="black")

display(pl)

# COMMAND ----------

# MAGIC %md The figure shows a histogram of component replacements divided into the four component types over the entire maintenance history. It looks like these four components are replaced at similar rates.
# MAGIC 
# MAGIC There are many ways we might want to look at this data including calculating how long each component type lasts, or the time history of component replacements within each machine. This will take some preprocess of the data, which we are delaying until we do the feature engineering steps in the next example notebook.
# MAGIC 
# MAGIC Next, we convert the errors data to a Spark dataframe, and verify the data types have converted correctly. 

# COMMAND ----------

# The data was read in using a Pandas data frame. We'll convert 
# it to pyspark to ensure it is in a Spark usable form for later 
# manipulations.
maint_spark = spark.createDataFrame(maint, 
                              verifySchema=False)

# We no longer need the pandas dataframe, so we can release that memory.
del maint

# Check data type conversions.
maint_spark.printSchema()

# COMMAND ----------

# MAGIC %md Now we write the spark dataframe to an Azure blob storage container for use in the remaining notebooks of this scenario.

# COMMAND ----------

# Write the Maintenance data set to intermediate storage
maint_spark.write.mode('overwrite').saveAsTable(MAINT_DATA)

print("Maintenance files saved!")

# COMMAND ----------

# MAGIC %md ### Telemetry data set
# MAGIC 
# MAGIC The telemetry time-series data consists of voltage, rotation, pressure, and vibration sensor measurements collected from each  machines in real time. The data is averaged over an hour and stored in the telemetry logs.

# COMMAND ----------

# Github has been having some timeout issues. This should fix the problem for this dataset.
import socket
socket.setdefaulttimeout(90)

# load raw data from the GitHub URL
datafile = "telemetry.csv"

# Download the file once, and only once.
if not os.path.isfile(datafile):
    urllib.request.urlretrieve(basedataurl+datafile, datafile)
    
# Read into pandas
telemetry = pd.read_csv(datafile, encoding='utf-8')

# handle missing values
# define groups of features 
features_datetime = ['datetime']
features_categorical = ['machineID']
features_numeric = list(set(telemetry.columns) - set(features_datetime) - set(features_categorical))

# Replace numeric NA with 0
telemetry[features_numeric] = telemetry[features_numeric].fillna(0)

# Replace categorical NA with 'Unknown'
telemetry[features_categorical]  = telemetry[features_categorical].fillna("Unknown")

# Counts...
print(telemetry.count())

# Examine 10 rowzs of data.
telemetry.head(10)

# COMMAND ----------

# Check the incoming schema, we want to convert datetime to the correct type.
# format datetime field which comes in as string
telemetry.dtypes

# COMMAND ----------

# MAGIC %md Rather than plot 8.7 million data points, this figure plots a month of measurements for a single machine. This is representative of each feature repeated for every machine over the entire year of sensor data collection.

# COMMAND ----------

plt_data = telemetry.loc[telemetry['machineID'] == 1]

# format datetime field which comes in as string
plt_data['datetime'] = pd.to_datetime(plt_data['datetime'], format="%Y-%m-%d %H:%M:%S")


# Quick plot to show structure
plot_df = plt_data.loc[(plt_data['datetime'] >= pd.to_datetime('2015-02-01')) &
                       (plt_data['datetime'] <= pd.to_datetime('2015-03-01'))]

plt_data = pd.melt(plot_df, id_vars=['datetime', 'machineID'])

pl = ggplot(aes(x="datetime", y="value", color = "variable", group="variable"), plt_data) +\
    geom_line() +\
    scale_x_date(labels=date_format('%m-%d')) +\
    facet_grid('variable', scales='free_y')

display(pl)

# COMMAND ----------

# MAGIC %md The figure shows one month worth of telemetry sensor data for one machine. Each sensor is shown in it's own panel.
# MAGIC 
# MAGIC Next, we convert the errors data to a Spark dataframe, and verify the data types have converted correctly. 

# COMMAND ----------

# The data was read in using a Pandas data frame. We'll convert 
# it to pyspark to ensure it is in a Spark usable form for later 
# manipulations.
# This line takes about 9.5 minutes to run.
telemetry_spark = spark.createDataFrame(telemetry, verifySchema=False)

# We no longer need the pandas dataframes, so we can release that memory.
del telemetry
del plt_data
del plot_df

# Check data type conversions.
telemetry_spark.printSchema()

# COMMAND ----------

# MAGIC %md Now we write the spark dataframe to an Azure blob storage container for use in the remaining notebooks of this scenario.

# COMMAND ----------

# Write the telemetry data set to intermediate storage
telemetry_spark.write.mode('overwrite').saveAsTable(TELEMETRY_DATA)

print("Telemetry files saved!")

# COMMAND ----------

# MAGIC %md ### Failures data set
# MAGIC 
# MAGIC Failures correspond to component replacements within the maintenance log. Each record contains the Machine ID, component type, and replacement datetime. These records will be used to create the machine learning labels we will be trying to predict.

# COMMAND ----------

# load raw data from the GitHub URL
datafile = "failures.csv"

# Download the file once, and only once.
if not os.path.isfile(datafile):
    urllib.request.urlretrieve(basedataurl+datafile, datafile)
    
# Read into pandas
failures = pd.read_csv(datafile, encoding='utf-8')

print(failures.count())
failures.head(10)

# COMMAND ----------

# MAGIC %md The following histogram details the distribution of the failure records obtained from failure log. This log was built originally from component replacements the maintenance log file. 

# COMMAND ----------

# Plot failures
pl = ggplot(aes(x="failure"), failures) + geom_bar(fill="blue", color="black")
display(pl)

# COMMAND ----------

# MAGIC %md The figure shows failure related replacements occured for each of the 4 component types over the entire year.
# MAGIC 
# MAGIC Next, we convert the maintenance data to PySpark and store it in an Azure blob.

# COMMAND ----------

# The data was read in using a Pandas data frame. We'll convert 
# it to pyspark to ensure it is in a Spark usable form for later 
# manipulations.
failures_spark = spark.createDataFrame(failures, 
                                       verifySchema=False)

# Check data type conversions.
failures_spark.printSchema()

# COMMAND ----------

# MAGIC %md Now we write the spark dataframe to an Azure blob storage container for use in the remaining notebooks of this scenario.

# COMMAND ----------

# Write the failures data set to intermediate storage
failures_spark.write.mode('overwrite').saveAsTable(FAILURE_DATA)

print("Failure files saved!")

# Time the notebook execution. 
# This will only make sense if you "Run All" cells
toc = time.time()
print("Full run took %.2f minutes" % ((toc - tic)/60))

# COMMAND ----------

# MAGIC %sh ls -l

# COMMAND ----------

# MAGIC %md ## Conclusion
# MAGIC 
# MAGIC We have now downloaded the required data files in csv format. We converted the data into Pandas data frames so we could generate a few graphs to help us understand what was in each data file. Then saved them into an Azure Blob storage container as Spark data frames for use in the remaining analysis steps. The `Code\2_feature_engineering.ipynb` Jupyter notebook will read these spark data frames from Azure blob and generate the modeling features for out predictive maintenance machine learning model.