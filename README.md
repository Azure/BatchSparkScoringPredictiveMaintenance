# Batch scoring of SPARK machine learning models 

## Overview

This scenario demonstrates batch scoring of a SPARK machine learning model on Azure Databricks. We use a predictive maintenance scenario where we classify machine sensor readings to classify a set of four machine components into healthy or unhealthy requiring maintenance states. The resulting supervised multi-class classifier model scores batches of new observations through a regularly scheduled Azure Databricks notebook jobs.

For this scenario, “Input Data” in the architecture diagram refers to a set of five simulated data sets related to realistic machine operating conditions. The solution uses methods from the PySpark MLlib machine learning library but the scoring process can be generalized to use any Python or R model hosted on Azure Databricks to make real-time predictions.

For an in depth description of the scenario, we have documented the operations in each of the supplied Jupyter notebooks contained in the `./notebooks/` directory of this repository.

## Design

This solution uses the Azure Databricks service. We create jobs that set up the batch scoring demonstration. Each job executes a Databricks notebook to prepare the data and create the full solution.

 1. Ingest process downloads the simulated data sets from a GitHub site and converts and stores them as Spark dataframes on the Databricks DBFS.

 2. Feature engineering transforms and combines the data sets into an analysis data set. The analysis data set can be targeted for training a model, or scoring data for a production pipeline. Each analysis data set is also stored in the Databricks DBFS.

 3. Training process takes a subset of the complete data and constructs a model we can use to predict future outcomes. The model is stored in the Databricks DBFS for use by the scoring notebook.

 4. The scoring process uses a different subset of the data, including data not yet collected to predict the current and future state of the machine. The model results are stored back onto the Databricks DBFS.

![Databricks Architecture diagram](./architecture.jpg "Architecture diagram")

# Prerequisites

We assume you have cloned the GitHub repository to your working compute instance (local computer or VM). The repository is located at: `https://github.com/Azure/BatchSparkScoringPredictiveMaintenance.git`

We will be using a Databricks command line utility (CLI) to automate many of these tasks. You should have a Python version installed. We require Python Version > 2.7.9 or > 3.6 for Databricks CLI requirements.

## Azure Databricks
This example is designed to run on Azure Databricks. You can provision the service through the Azure portal at:

https://ms.portal.azure.com/#create/Microsoft.Databricks

This example will run on the Standard pricing tier. 

## Databricks cluster

Once your Azure Databricks instance has been deployed, you will need to create a compute cluster to execute the notebooks. Launch your new workspace from the Azure portal, select the *Clusters* icon, and `Create Cluster` to provision a new cluster with Python Version 3. The remaining defaults values are acceptable.

## Databricks CLI

Installing the Databricks CLI will simplify some of the operations required for this scenario. The first step is to import the Jupyter 10 notebooks from the repository into your Databricks workspace. This can be accomplished with 1 command once the CLI is connected to your Azure Databricks instance. The Databricks CLI available here:

https://github.com/databricks/databricks-cli

From a command line, you can pip install the CLI using 

`pip install --upgrade databricks-cli`

# Setup

 * Clone the repo 
 
 ```
 git clone https://github.com/Azure/BatchSparkScoringPredictiveMaintenance
 ```

 * cd into the root directory of your cloned repo

The next two subsections of this document detail how to:

 * Connect the CLI to your Databricks instance to simplify the import of repo notebooks.
 * Import the repo notebooks into your Databricks workspace

## Connect the CLI to your Databricks instance

We need to connect the CLI to your databricks instance. This can be done using a Databricks generated [Authentication token](https://docs.databricks.com/api/latest/authentication.html#token-management).

This operation will connect the CLI to this Databricks instance for all commands that follow. 

Start from a command line, using the following command:

`databricks configure --token`

This will prompt you for your Azure Databricks hostname, which is the url portion of the web address from your browser. In `eastus` region, it will be `https://eastus.azuredatabricks.net/`. You will not use the POST arguments for the hostname (everything including and following the '?' character). You will also need to create and copy an [Authentication token](https://docs.databricks.com/api/latest/authentication.html#token-management). Instructions are provided at the link.

## Import Notebooks

Next, use the CLI to copy the scenario notebooks to your Databricks instance with the following command.

`databricks workspace import_dir [OPTIONS] SOURCE_PATH TARGET_PATH`

Change into the local copy of the repository. Your `SOURCE_PATH` will be the `./notebooks` directory. The target path will include your user name, which you can get from the Azure Databricks UI, it should be related to your Azure AD email of the form `<uname@example.com>`. The `[TARGET_PATH]` will then be of the form `/Users/<uname@example.com>/notebooks`. 

The command should look like the following:

`databricks workspace import_dir notebooks /Users/<uname@example.com>/notebooks`

This will copy all required notebooks into the `notebooks` folder of your Azure Databricks Workspace. 

# Steps

To create the full example scenario, run through the following notebooks now located in your Azure Databricks workspace.

  * [Ingest Data](https://github.com/Azure/BatchSparkScoringPredictiveMaintenance/blob/master/notebooks/1_data_ingestion.ipynb) in your `notebooks/1_data_ingestion` notebook.
  * [Model Training Pipeline](https://github.com/ehrlinger/BatchSparkScoringPredictiveMaintenance/blob/master/notebooks/2_Training_Pipeline.ipynb) in your `notebooks/2_Training_Pipeline` notebook.
  * [Data Scoring Pipeline](https://github.com/ehrlinger/BatchSparkScoringPredictiveMaintenance/blob/master/notebooks/3_Scoring_Pipeline.ipynb)  in your `notebooks/3_Scoring_Pipeline` notebook
  * (optional) [Create a batch scoring Databricks Job](https://github.com/ehrlinger/BatchSparkScoringPredictiveMaintenance/blob/master/BatchScoringJob.md) using the Databricks CLI and the instruction at the link.

This scenario demonstrates how to automate the batch scoring of a predictive maintenance solution. The batch process is executed through Databricks Jobs, which automate running Databricks notebooks either on demand or on a schedule.

# Cleaning up

The easiest way to cleanup this work is to delete the Azure Databricks instance through the Azure portal (https://portal.azure.com).

# References

This scenario has been developed using a similar predictive maintenance use case published at following reference locations:

 * [Predictive Maintenance Solution Template](https://docs.microsoft.com/en-us/azure/machine-learning/team-data-science-process/cortana-analytics-playbook-predictive-maintenance)
 * [Predictive Maintenance Modeling Guide](https://gallery.azure.ai/Collection/Predictive-Maintenance-Modelling-Guide-1)
 * [Predictive Maintenance Modeling Guide using SQL R Services](https://gallery.azure.ai/Tutorial/Predictive-Maintenance-Modeling-Guide-using-SQL-R-Services-1)
 * [Predictive Maintenance Modeling Guide Python Notebook](https://gallery.azure.ai/Notebook/Predictive-Maintenance-Modelling-Guide-Python-Notebook-1)
 * [Predictive Maintenance using PySpark](https://gallery.azure.ai/Tutorial/Predictive-Maintenance-using-PySpark)
 * [Predictive Maintenance scenario](https://docs.microsoft.com/en-us/azure/machine-learning/desktop-workbench/scenario-predictive-maintenance)
 * [Deep learning for predictive maintenance](https://docs.microsoft.com/en-us/azure/machine-learning/desktop-workbench/scenario-deep-learning-for-predictive-maintenance)

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

### Author: 
John Ehrlinger <john.ehrlinger@microsoft.com>
