# Create a batch scoring Databricks Job

The batch scoring process is actually a pipeline of operations. We assume raw data arrives through the some process. The scoring operation has to transform the raw data into the correct format for the model to consume, then the model makes predictions which we store for later consumption. 

We've created the `notebooks/3_Scoring_Pipeline.ipynb` to execute a full scoring pipeline. The notebook takes a `start_date` and `to_date`, as well as a `model` to indicate which model to use and a `results_table` name to store the model predictions. The notebook runs the `notebooks/2a_feature_engineering.ipynb` notebook to transform the data, storing the scoring data in a temporary table and then runs the `notebooks/3a_model_scoring.ipynb` notebook with the specified model and results data target data set to store the model predictions. 

## Customize the JSON templates

You can create Databricks jobs through the Azure Databricks workspace portal or through the Databricks CLI. We demonstrate using the CLI and have provided a set of Job template files. Before using these templates, you will need to provide the information to connect to your specific Databricks instance. 

1. The Databricks Cluster ID you want to run the job on.

You can only get the ClusterID from the Databricks CLI. Assuming you have already connected the CLI to your Databricks instance, run the following command from a terminal window to get the cluster ID from the first field of the resulting table.

```
databricks clusters list
```

2. The workspace username containing the notebooks to run.

This will be the same username you used to copy the notebooks to your workspace and should take the form of `username@example.com`.


Using this information, we have provided a script to customize the templates for connecting to your Databricks cluster. Execute the following command from a terminal window, in your repository root directory.

```
python scripts/config.py [-h] [-c CLUSTERID] [-u USERNAME] ./jobs/
```

This command reads the `./jobs/*.tmpl` files, replaces the clusterID and username placeholder strings with the specifics for your Databricks cluster, storing the modified files to the JSON jobs scripts we'll use to setup and demonstrate the Batch scoring processes.


## Create the batch scoring job

Create the scoring pipeline job using the CLI command:

`databricks jobs create --json-file jobs/03_CreateScoringPipeline.json`

This particular batch job is configured to only run on demand as the previous jobs. However, adding a _schedule_ command to the JSON file in `jobs/03_CreateScoringPipeline.json`.

```
"schedule": {
    "quartz_cron_expression": "0 15 22 ? * *",
    "timezone_id": "America/Los_Angeles"
  },
```

Details to customize this scheduler can be found in the documentation at (https://docs.databricks.com/api/latest/jobs.html#create)

Run the job with default parameters as before:

`databricks jobs run-now --job-id <jobID>`

To specify different parameters, use the following call.
```
databricks jobs run-now --job-id <jobID> --notebook-params { "results_data":"predictions","model":"RandomForest","start_date":"2015-11-15","to_date":"2017-01-01"}
```


on Windows, we need to escape out the quote characters.
```
databricks jobs run-now --job-id <jobID> --notebook-params {\"start_date\":\"2015-11-15\",\"to_date\":\"2017-01-01\", \"results_data\":\"predictions", \"model\":\"RandomForest\"}
```

The entire workflow job will take about 2-3 minutes to complete given this 2.5 months of data.