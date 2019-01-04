# Batch scoring of SPARK machine learning models 

## Overview

This scenario demonstrates batch scoring of a SPARK machine learning model on Azure Databricks. We use a predictive maintenance scenario where we classify machine sensor readings to classify a set of four machine components into _healthy_ or _unhealthy requiring maintenance_ states. The resulting supervised multi-class classifier model scores batches of new observations through a regularly scheduled Azure Databricks notebook jobs.

The solution uses methods from the PySpark MLlib machine learning library but the scoring process can be generalized to use any Python or R model hosted on Azure Databricks to make real-time predictions.

For an in depth description of the scenario, we have documented the operations in each of the supplied Jupyter notebooks contained in the `./notebooks/` directory of this repository.

## Design

This solution uses the Azure Databricks service. We create jobs that set up the batch scoring demonstration. Each job executes a Databricks notebook to prepare the data and create the full solution.

 1. Ingest process downloads the simulated data sets from a GitHub site and converts and stores them as Spark dataframes on the Databricks DBFS. “Input Data” in the architecture diagram refers to a set of five simulated data sets related to realistic machine operating conditions. 

 2. Feature engineering transforms and combines the data sets into an analysis data set. The analysis data set can be targeted for training a model, or scoring data for a production pipeline. Each analysis data set is also stored in the Databricks DBFS.

 3. Training process takes a subset of the complete data and constructs a model we can use to predict future outcomes. The model is stored in the Databricks DBFS for use by the scoring notebook.

 4. The scoring process uses a different subset of the data, including data not yet collected to predict the current and future state of the machine. The model results are stored back onto the Databricks DBFS.

![Databricks Architecture diagram](./architecture.jpg "Architecture diagram")

# Prerequisites

We assume you have an Azure subscription. You will also need access to git on your working compute instance (local computer or VM). The repository is located at: `https://github.com/Azure/BatchSparkScoringPredictiveMaintenance`

 We also require Python Version > 2.7.9 or > 3.6 as specified for using the Databricks CLI.

## Azure Databricks

This example is designed to run on Azure Databricks. You can provision the service through the Azure portal at:

https://ms.portal.azure.com/#create/Microsoft.Databricks

This example will run on the Standard pricing tier.



## Databricks cluster

Once your Azure Databricks instance has been deployed, you will need to create a compute cluster to execute the notebooks. Launch your new workspace from the Azure portal, select the *Clusters* icon, and `Create Cluster` to provision a new cluster with Python Version 3. The remaining defaults values are acceptable.

## Databricks CLI

We will be using a Databricks command line utility (CLI) to automate running notebook tasks using the Databricks Jobs construct. Installing the Databricks CLI will simplify some of the operations required for this scenario. The first step is to import the Jupyter notebooks from the repository into your Databricks workspace. This can be accomplished with 1 command once the CLI is connected to your Azure Databricks instance.

From a command line, you can pip install the CLI using 

```
pip install --upgrade databricks-cli
```

# Setup

 * Clone the GitHub repository: 
 
 ```
 git clone https://github.com/ehrlinger/BatchSparkScoringPredictiveMaintenance.git
 ```

 * cd into the root directory of your cloned repository

The next two subsections of this document detail how to:

 * Connect the CLI to your Databricks instance to simplify the import of repo notebooks.
 * Import the repo notebooks into your Databricks workspace

## Connect the CLI to your Databricks instance

We need to connect the CLI to your databricks instance. This can be done using a Databricks generated [Authentication token](https://docs.databricks.com/api/latest/authentication.html#token-management). This linking only needs to be done once.


 1. Copy the url portion of the web address of your Azure Databricks instance from your browser. You will not use the POST arguments for the hostname (everything including and following the '?' character). In `eastus` region, it will be `https://eastus.azuredatabricks.net/`.

 2. Create and copy a an authentication token. Instructions are provided at the link above.

 3. From your working machine command line, `databricks configure --token`. This will prompt you for your Azure Databricks hostname and the authentication token.

## Import Notebooks

Use the CLI to copy the scenario notebooks to your Databricks instance. From your working machine command line, change into the local copy of the repository.  Then `databricks workspace import_dir [OPTIONS] SOURCE_PATH TARGET_PATH` 

  * The `SOURCE_PATH` will be the `./notebooks` directory. 
  * The `TARGET_PATH` will include your user name, which you can get from the Azure Databricks UI, it should be related to your Azure AD email of the form `<uname@example.com>`.  The whole `[TARGET_PATH]` should be of the form `/Users/<uname@example.com>/notebooks`. 

The command should look like the following:

`databricks workspace import_dir notebooks /Users/<uname@example.com>/notebooks`

This will copy all required notebooks into the `notebooks` folder of your Azure Databricks Workspace. 

# Steps

To create the full example scenario, through your Azure Databricks workspace, run through the following notebooks now located in your Azure Databricks workspace. 

When running the notebooks, you may have to start your Azure Databricks cluster or attach these notebooks to your Azure Databricks cluster. The UI will prompt you if this is required.

  * [Ingest Data](https://github.com/Azure/BatchSparkScoringPredictiveMaintenance/blob/master/notebooks/1_data_ingestion.ipynb) Run all cells in the `notebooks/1_data_ingestion` notebook on the Azure Databricks workspace.
  * [Model Training Pipeline](https://github.com/ehrlinger/BatchSparkScoringPredictiveMaintenance/blob/master/notebooks/2_Training_Pipeline.ipynb) Run all cells in the `notebooks/2_Training_Pipeline` notebook on the Azure Databricks workspace.
  * [Data Scoring Pipeline](https://github.com/ehrlinger/BatchSparkScoringPredictiveMaintenance/blob/master/notebooks/3_Scoring_Pipeline.ipynb) Run all cells in the `notebooks/3_Scoring_Pipeline` notebook on the Azure Databricks workspace.
  * (optional) Instruction to [create a batch scoring Databricks Job](https://github.com/ehrlinger/BatchSparkScoringPredictiveMaintenance/blob/master/BatchScoringJob.md) using the Databricks CLI are documented at the link.

This scenario demonstrates how to automate the batch scoring of a predictive maintenance solution. The batch process is executed through Databricks Jobs, which automate running Databricks notebooks either on demand or on a schedule.

# Cleaning up

The easiest way to cleanup this work is to delete the resource group containing the Azure Databricks instance.

  1. Through the Azure portal (https://portal.azure.com) search for `databricks`. 
  1. Locate and delete the resource group containing the Azure Databricks instance. This will remove the cluster, Databricks instance which includes the notebooks and data artifacts used in this scenario.

You may also want to remove the Databricks CLI from your python environment with
```
pip uninstall databricks-cli
```

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
