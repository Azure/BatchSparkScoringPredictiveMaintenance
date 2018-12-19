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

This particular example will run on the Standard pricing tier. 

## Databricks cluster

Once your Azure Databricks instance has been deployed, we can create a compute cluster. Launch your new workspace, select the *Clusters* icon. and Create a new cluster with Python Version 3.

## Databricks CLI

In order to avoid trying to explain how to execute the required steps in the Databricks UI, we will require the Databricks CLI available here:

https://github.com/databricks/databricks-cli

From a command line, you can pip install using 

`pip install --upgrade databricks-cli`

# Setup

Before running the notebooks, we need to setup the environment.

 * Connect the CLI to your Databricks instance
 * Import the repo notebooks into your Databricks workspace
 * Get the clusterID to direct the execution compute context
 * Customize the Jobs JSON scripts to direct Jobs to the correct Databricks Workspace and ClusterID locations. 


## Connect the CLI

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

## Get cluster Id

We also need the Databricks ClusterID to direct all compute operations on your Databricks Instance. The only way to find the ClusterID is through the following CLI command:

`databricks clusters list`

The clusterID is in the first field of the list. We will use this to point the Databricks Jobs to execute on a specific execute cluster.

## Customize the jobs scripts

This scenario is built using a series of Jupyter notebooks, executed using the Databricks Jobs construct. We can construct the Databricks Jobs using mouse clicks through the Azure Databricks UI or through commands on the Databricks CLI. We've included JSON script templates to create Databricks Jobs using the CLI. The `config.py` will customize these scripts to connect to your specific Azure Databricks instance. To connect, we need your username (`<uname@example.com>`) and the cluster ID (`<clusterID>`) found above.

From the root repository directory, the `config.py` command usage:

```
python scripts/config.py [-h] [-c CLUSTERID] [-u USERNAME] ./jobs/
```

This command reads the `./jobs/*.tmpl` files, replaces the clusterID and username placeholder strings with the specifics for your Databricks cluster, storing the modified files to the JSON jobs scripts we'll use to setup and demonstrate the Batch scoring processes.

# Steps

Before we can get to scoring a machine learning model, we need to go through the steps of a data science process. 

1. **Ingest the data**
1. **Manipulate and Transform the data** into an analysis data set
1. **Create a model**
1. **Score** new observations with the created model

For each of these steps, we will use the Databricks CLI to create a [Databricks Job](https://docs.databricks.com/user-guide/jobs.html). Each Databricks Job will automate the running of a Databricks notebook either immediately or on a scheduled basis. We create the jobs using instructions contained in JSON files located in the `./jobs/` folder of this repository.

This is a serial process, where each step in the process depends on data artifacts created in the previous step. Customization of this scenario will require a similar set of data artifacts.

## Ingest data

The first step is to download the raw data sets and store them as Spark Dataframes accessible to your Azure Databricks instance. The actual process is done through the `1_data_ingestion.ipynb` Jupyter notebook that we copied into the `notebooks` folder of your workspace. You can either use the Azure Databricks UI to connect the notebook to your cluster and execute all cells, or use the CLI to create a Databricks Job to do the same process automatically. We'll use the Job script we customized above.

To create a job from the CLI, use the following command:
`databricks jobs create --json-file jobs/01_CreateDataIngestion.json`

This registers the job, and describes what notebook to execute on which cluster. Successful completion of the command returns a `<jobID>` number we can use in subsequent commands. To run the job, use the `<jobID>` in the following command:

`databricks jobs run-now --job-id <jobID>`

You can review the registered jobs in your Azure Databricks instance through the UI or with the CLI command: 

`databricks jobs list`

To check on specific state of submitted job runs, you can use the command:

`databricks runs list`

The first run may take some time, as it will liekly need to start the target Databricks cluster before running all cells in the data ingestion notebook. If the cluster is already running, this job typically will take about 8-10 minutes to complete. 

To examine the SPARK data frames constructed in the  `notebooks\1_data_ingestion.ipynb` execution, an additional notebook has been included in the repostiory and copied to your Azure Databricks Workspace.  The `notebooks\1a_raw_data_exploring.ipynb` notebook can be run interactively through the Azure Databricks Workspace UI. You must run the ingest data job for before running the exploration notebook cells. The exploration notebook details the simulated data sets we used for this predictive maintenance solution example.

## Manipulate and Transform the data

Once the data is in place, we want to create analysis data sets. The manipulations and transformation used to create the training set to build the model, should be reused to test and calibrate the model and again reused on incoming production data to score new observations. 

For this scenario, we use a temporal splitting strategy. We train the model on all data collected before October 30, 2015, and use the remaining data to simulate production data for scoring. The `notebooks/2_feature_engineering.ipynb` notebook uses [Databricks Input widgets] to allow input parameters for specifying the output storage data set (`training_data` by default) and the start date (`2000-01-01`) and end date (`2015-10-30`) of the job run. 

We create the feature engineering job using the following command:

`databricks jobs create --json-file jobs/02_CreateFeatureEngineering.json`

Again, we run the job specifying the returned jobID.

`databricks jobs run-now --job-id <jobID>`

To change the job parameters, we can over ride the default parameters built into the notebook using the `--notebook-params` command argument.

```
databricks jobs run-now --job-id <jobID> --notebook-params {"FEATURES_TABLE":"testing_data","Start_Date":"2015-11-15","zEnd_Date":"2017-01-01"}
```

On windows command line, we need to escape the double quotes for this to work: 

`
```
databricks jobs run-now --job-id <jobID> --notebook-params {\"FEATURES_TABLE\":\"testing_data\",\"Start_Date\":\"2015-11-15\",\"zEnd_Date\":\"2017-01-01\"}
```

The data manipulation and transformation job should take under 3 minutes to build the training data if your cluster is already started. Smaller date ranges may run faster.

We again provide an data exploration notebook `notebooks\2a_feature_exploration.ipynb` as an example. This notebook details some of the feature engineering steps we used for this predictive maintenance solution example.

## Create a model

Once we have a training set, we can build our model. We build the model in the `notebooks\3_model_building.ipynb` notebook, which also persists the model in a `pkl` file to the Databricks file system. 

We create the model building job from the command line:
`databricks jobs create --json-file jobs/03_CreateModelBuilding.json`

and use the same `run-now` CLI command to run the job.
`databricks jobs run-now --job-id <jobID>`

By default the model building job builds a RandomForest spark model using the `training_data` data set we created in the `02_CreateFeatureEngineering` job. We have included parameters to change to a DecisionTree model and to change the `TRAINING_TABLE` data set name as well. 

```
databricks jobs run-now --job-id <jobID> --notebook-params {\"model\":\"DecisionTree\"}
```

The model is stored to the Databricks files system at `dbfs:/storage/models/model.pqt`. If you already have a SPARK model saved in the parquet format `(.pqt)` format, you can copy using the CLI command `dbfs cp <SRC> <DST>`.

`dbfs cp  -r model.pqt dbfs:/storage/models/model.pqt`

In order to bring your own model, you'll also need to bring your own data (1. data ingestion) and your own manipulation/transformation (2. feature engineering) processes before being able to score new data with it (4. score).

To get an idea how the model performs, the `notebooks\3a_model_testing.ipynb` notebook loads the stored model and a feature data set specified in the notebook parameter fields, and calculates a set of model metrics.

It will take about 4-5 minutes to train the SPARK RandomForest model.

## Score new observations

To here, we all previous jobs/notebooks have been geared toward setting up the batch scoring operation. The scoring process requires new data observations to be manipulated and transformed just as the training data in order for the model to predict the target. In this scenario, we apply the feature engineering notebook (`notebooks/02_feature_engineering.ipynb`) to new data that we assume was ingested into the same raw data sets that the training data was stored in. 

We simulate this process by selecting a new date range from the simulation data we ingested in the first data ingestion job. There are data leakage issues possible in predictive maintenance scenarios if we select data starting immediately after the training data. This is do to the time series lagging feature engineering transform required to inform the model of past machine behavior. For this reason, we start the scoring data two months after the last observation in the training data. The window gap can be shortened to the feature lag length, though we did not do that here.

Step one is to run the feature engineering step with a new `FEATURES_TABLE` (`scoring_inpout`), and a new start date `2015-12-30` and end date `2016-04-30`:

```
databricks jobs run-now --job-id <jobID> --notebook-params {\"FEATURES_TABLE\":\"scoring_input\",\"Start_Date\":\"2015-12-30\",\"zEnd_Date\":\"2016-04-30\"}
```

The `notebooks\4_model_scoring.ipynb` notebook takes the scoring data set from the `SCORING_DATA` data set, and scores the observations using the `model_type` model, storing the model prediction results in the `RESULTS_DATA` data set

Create the job using the CLI command:

`databricks jobs create --json-file jobs/04_CreateModelScoring.json`

Remember to run the feature engineering job above before running this job with default parameters (that point to the `scoring_input` data set) as before:

`databricks jobs run-now --job-id <jobID>`

Again, you can manipulate the notebook parameters for `SCORING_DATA` set, the `MODEL_TYPE` (`RandomForest` or `DecisionTree`) to indicate where the model is stored. The notebook looks for the model in the `dbfs:/storage/models/[MODEL_TYPE].pqt"` file. The notebook then stores the observation scores in the `RESULTS_TABLE` for post processing and final consumption.
  
You can check on your run status using the CLI command:

`databricks runs list`

This allows you to check on which jobs have been run. If your scoring job fails, make sure you've run the feature engineering job to create the `scoring_input` data set.

The scoring job will take less than 1 minute. 

# Batch scoring job

Notice from the previous section that the scoring process is actually a pipeline of operations. We assume raw data arrives through the some process. The scoring operation has to transform the raw data into the correct format for the model to consume, then the model makes predictions which we store for later consumption. 

We've created the `notebooks/05_full_scoring_workflow.ipynb` to execute a full scoring pipeline. The notebook takes a `start_date` and `end_date`, as well as a `model_type` to indicate which model to use and a `results_table` name to store the model predictions. The notebook runs the `notebooks/2_feature_engineering.ipynb` notebook to transform the data, storing the results in the `HPscoring_input` data set, and then runs the `notebooks/4_model_scoring.ipynb` notebook with the specified model and results data target data set. 

Create the scoring workflow job using the CLI command:

`databricks jobs create --json-file jobs/05_CreateScoringWorkflow.json`

This particular batch job is configured to only run on demand as the previous jobs. However, adding a _schedule_ command to the JSON file in `jobs/05_CreateScoringWorkflow.json`.

```
"schedule": {
    "quartz_cron_expression": "0 15 22 ? * *",
    "timezone_id": "America/Los_Angeles"
  },
```

Details can be found in the documentation at (https://docs.databricks.com/api/latest/jobs.html#create)

Run the job with default parameters as before:

`databricks jobs run-now --job-id <jobID>`

To specify different parameters, use the following call:

```
databricks jobs run-now --job-id <jobID> --notebook-params {\"Start_Date\":\"2015-11-15\",\"zEnd_Date\":\"2017-01-01\", \"RESULTS_DATA\":\"scored_data", \"MODEL_TYPE\":\"DecisionTree\"}
```

The entire workflow job will take about 2-3 minutes to complete given this 2.5 months of data.

# Conclusion

This scenario demonstrates how to automate the batch scoring of a predictive maintenance solution. The batch process is executed through Databricks Jobs, which automate running Databricks notebooks either on demand or on a schedule.

In a production setting for this scenario, we would expect data to arrive from different machines asynchronously. A regularly scheduled batch job could query the data store for machine ID's that have new data since the last scoring batch job. A query would then pull the scoring data window corresponding to the feature engineering process, score the data, and append these results into the data storage for use by the end user consumption process.

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
