

# Batch scoring of SPARK machine learning models 

## Overview

This scenario demonstrates batch scoring of a SPARK machine learning model on Azure Databricks. The model is constructed for a predictive maintenance scenario where we classify machine sensor readings into healthy or unhealthy and requiring maintenance for a set of four machine components. The resulting supervised multi-class model scores batches of new observations through a regularly scheduled Azure Databricks notebook tasks.

For this scenario, “Input Data” in the architecture diagram refers to a set of five simulated data sets related to realistic machine operating conditions. The solution uses methods from the PySpark MLlib machine learning library but the scoring process can be generalized to use any Python or R model hosted on Azure Databricks to make real-time predictions.

## Design

This solution uses the Azure Databricks service. We create jobs that set up the batch scoring demonstration. Each job executes a Databricks notebook to prepare the data and create the full solution.

 1. Ingest process downloads the simulated datasets from a GitHub site and converts and stores them as Spark dataframes on the Databricks DBFS.

 2. Feature engineering transorms and combines the data sets into an analysis dataset. The analysis data set can be targeted for training a model, or scoring data for a production pipeline. Each analysis dataset is also stored in the Databricks DBFS.

 3. Training process takes a subset of the complete data and constructs a model we can use to predict future outcomes. The model is stored in the Databricks DBFS for use by the scoring notebook.

 4. The scoring process uses a different subset of the data, including data not yet collected to predict the current and future state of the machine. The model results are stored back onto the Databricks DBFS.

![Databricks Architecture diagram](./architecture.jpg "Architecture diagram")

# Prerequisites

We will be using a command line on your computer. You should have a Python version installed. We require Python Version > 2.7.9 or > 3.6 because of the Databricks CLI requirements.

## Azure Databricks
This example is designed to run on Azure Databricks. You can provision the service through the Azure portal at:

https://ms.portal.azure.com/#create/Microsoft.Databricks

This particular example will run on the Standard pricing tier. 

## Databricks cluster

Once your Azure Databricks instance has been deployed, we can create a compute cluster. Launch your new workspace, select the *Clusters* icon. and Create a new cluster with Python Version 3.

## Databricks CLI

In order to avoid trying to explain how to execute the required steps in the Databricks UI, we will require the Databricks CLI available here:

https://github.com/databricks/databricks-cli

From a command line, you can pip install using 

`pip install --upgrade databricks-cli`

Then connect the CLI to your databricks instance using your [Authentication token](https://docs.databricks.com/api/latest/authentication.html#token-management).

Use the following command:

`databricks configure --token`

This will prompt you for your Azure Databricks hostname (copy this from the browser address bar), and then the [Authentication token](https://docs.databricks.com/api/latest/authentication.html#token-management).

# Setup
dapid77c1b1ac815491a955735ec605376b5


## Import Notebooks

`databricks workspace import_dir notebooks /Users/<uname@example.com>/notebooks`

## Get cluster Id

`databricks clusters list`

## Setup databricks jobs 

### Ingest data

`databricks jobs create --json-file jobs/01_CreateDataIngestion.json`

`databricks jobs run-now --job-id <jobID>`

### Feature engineering

`databricks jobs create --json-file jobs/02_CreateFeatureEngineering.json`

`databricks jobs run-now --job-id <jobID>`

We supply parameters using the `--notebook-params` command.

`databricks jobs run-now --job-id <jobID> --notebook-params {"FEATURES_TABLE":"testing_data","Start_Date":"2015-11-15","zEnd_Date":"2017-01-01"}`

On windows command line, we need to escape the double quotes:

`databricks jobs run-now --job-id <jobID> --notebook-params {\"FEATURES_TABLE\":\"testing_data\",\"Start_Date\":\"2015-11-15\",\"zEnd_Date\":\"2017-01-01\"}`

### Create the model

`databricks jobs create --json-file jobs/03_CreateModelBuilding.json`

`databricks jobs run-now --job-id <jobID>`

`databricks jobs run-now --job-id <jobID> --notebook-params {\"model\":\"DecisionTree\"}`

If you already have a SPARK model saved in Parquet format, you can copy using the CLI command `dbfs cp <SRC> <DST>`.

`dbfs cp  -r model.pqt dbfs:/storage/models/model.pqt`

## Load the scoring job

We need to create the dataset we'll score

`databricks jobs run-now --job-id <jobID> --notebook-params {\"FEATURES_TABLE\":\"scoring_input\",\"Start_Date\":\"2015-12-30\",\"zEnd_Date\":\"2016-04-30\"}`

The load the scoring job

`databricks jobs create --json-file jobs/04_CreateModelScoring.json`

Then run the job.

`databricks jobs run-now --job-id <jobID>`

# Steps

Instructions on where to go (first notebook or folder)

# Cleaning up

Where applicable, what does the user have to manually scrub to clean it.

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

### Author: 
John Ehrlinger <john.ehrlinger@microsoft.com>
