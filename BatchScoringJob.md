# Create a batch scoring Databricks Job

The batch scoring process is actually a pipeline of operations. We assume raw data arrives through the some process. The scoring operation has to transform the raw data into the correct format for the model to consume, then the model makes predictions which we store for later consumption. 

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