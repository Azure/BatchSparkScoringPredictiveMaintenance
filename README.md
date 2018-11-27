# Overview

What will you actually learn? What is the problem that is being shown as an example? Who is the main audience? How long will it take a user to follow through, to it’s conclusion, this path?

# Design

The 10K foot view of the architecture that includes things like (DockerHub/VM/etc) to give a visual on what the overall design is.

This should also include supported platforms.

# Prerequisites

What do you need to have at your disposal?

## Databricks cluster
https://ms.portal.azure.com/#create/Microsoft.Databricks

## Databricsk CLI
https://github.com/databricks/databricks-cli

# Setup

Login to databricks CLI. 

## Import Notebooks

`databricks workspace import_dir notebooks /Users/example@databricks.com/notebooks`

## Setup databricks jobs 
`databricks jobs create --json-file jobs/01_CreateDataIngestion.json`
`databricks jobs run-now --job-id <jobID>`

`databricks jobs create --json-file jobs/02_FeatureEngineering.json`
`databricks jobs run-now --job-id <jobID>`

`databricks jobs create --json-file jobs/03_ModelBuilding.json`
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
