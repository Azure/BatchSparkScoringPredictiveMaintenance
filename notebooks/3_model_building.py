# Databricks notebook source
# MAGIC %md # Step 3: Model Building
# MAGIC 
# MAGIC Using the labeled feature data set constructed in the `Code/2_feature_engineering.ipynb` Jupyter notebook, this notebook loads the data from the Azure Blob container and splits it into a training and test data set. We then build a machine learning model (a decision tree classifier or a random forest classifier) to predict when different components within our machine population will fail. We store the better performing model for deployment in an Azure web service in the next. We will prepare and build the web service in the `Code/4_operationalization.ipynb` Jupyter notebook.
# MAGIC 
# MAGIC **Note:** This notebook will take about 2-4 minutes to execute all cells, depending on the compute configuration you have setup. 

# COMMAND ----------

# import the libraries
import os
import glob
import time

# for creating pipelines and model
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, VectorIndexer
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.sql.functions import col
from pyspark.sql import SparkSession

# For some data handling
import pandas as pd
import numpy as np

# Time the notebook execution. 
# This will only make sense if you "Run all cells"
tic = time.time()

# This is the final feature data file.
TRAINING_TABLE = 'training_table'
TESTING_TABLE = 'testing_table'
model_type = 'RandomForest' # Use 'DecisionTree' or 'GBTClassifier' or 'RandomForest'




# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.widgets.text("TRAINING_TABLE",TRAINING_TABLE)

dbutils.widgets.text("TESTING_TABLE",TESTING_TABLE)
dbutils.widgets.text("model", model_type)

# COMMAND ----------

# MAGIC %md Load the data and dump a short summary of the resulting DataFrame.

# COMMAND ----------

train_data = spark.sql("SELECT * FROM " + dbutils.widgets.get("TRAINING_TABLE"))
train_data.limit(10).toPandas().head(10)

# COMMAND ----------

type(train_data)

# COMMAND ----------

# MAGIC %md # Prepare the Training/Testing data

# COMMAND ----------

# MAGIC %md A fundamental practice in machine learning is to calibrate and test your model parameters on data that has not been used to train the model. Evaluation of the model requires splitting the available data into a training portion, a calibration portion and an evaluation portion. Typically, 80% of data is used to train the model and 10% each to calibrate any parameter selection and evaluate your model.
# MAGIC 
# MAGIC In general random splitting can be used, but since time series data have an inherent correlation between observations. For predictive maintenance problems, a time-dependent spliting strategy is often a better approach to estimate performance. For a time-dependent split, a single point in time is chosen, the model is trained on examples up to that point in time, and validated on the examples after that point. This simulates training on current data and score data collected in the future data after the splitting point is not known. However, care must be taken on labels near the split point. In this case, feature records within 7 days of the split point can not be labeled as a failure, since that is unobserved data. 
# MAGIC 
# MAGIC In the following code blocks, we split the data at a single point to train and evaluate this model. 

# COMMAND ----------

# define list of input columns for downstream modeling

# We'll use the known label, and key variables.
label_var = ['label_e']
key_cols =['machineID','dt_truncated']

# Then get the remaing feature names from the data
input_features = train_data.columns

# We'll use the known label, key variables and 
# a few extra columns we won't need.
remove_names = label_var + key_cols + ['failure','model_encoded','model' ]

# Remove the extra names if that are in the input_features list
input_features = [x for x in input_features if x not in set(remove_names)]

input_features

# COMMAND ----------

# MAGIC %md Spark models require a vectorized data frame. We transform the dataset here and then split the data into a training and test set. We use this split data to train the model on 9 months of data (training data), and evaluate on the remaining 3 months (test data) going forward.

# COMMAND ----------

# assemble features
va = VectorAssembler(inputCols=(input_features), outputCol='features')
train_data = va.transform(train_data).select('machineID','dt_truncated','label_e','features')

# set maxCategories so features with > 10 distinct values are treated as continuous.
featureIndexer = VectorIndexer(inputCol="features", 
                               outputCol="indexedFeatures", 
                               maxCategories=10).fit(train_data)

# fit on whole dataset to include all labels in index
labelIndexer = StringIndexer(inputCol="label_e", outputCol="indexedLabel").fit(train_data)

training = train_data

print(training.count())

# COMMAND ----------

# MAGIC %md # Classification models
# MAGIC 
# MAGIC A particualar troubling behavior in predictive maintenance is machine failures are usually rare occurrences compared to normal operation. This is fortunate for the business as maintenance and saftey issues are few, but causes an imbalance in the label distribution. This imbalance leads to poor performance as algorithms tend to classify majority class examples at the expense of minority class, since the total misclassification error is much improved when majority class is labeled correctly. This causes low recall or precision rates, although accuracy can be high. It becomes a larger problem when the cost of false alarms is very high. To help with this problem, sampling techniques such as oversampling of the minority examples can be used. These methods are not covered in this notebook. Because of this, it is also important to look at evaluation metrics other than accuracy alone.
# MAGIC 
# MAGIC We will build and compare two different classification model approaches:
# MAGIC 
# MAGIC  - **Decision Tree Classifier**: Decision trees and their ensembles are popular methods for the machine learning tasks of classification and regression. Decision trees are widely used since they are easy to interpret, handle categorical features, extend to the multiclass classification setting, do not require feature scaling, and are able to capture non-linearities and feature interactions.
# MAGIC 
# MAGIC  - **Random Forest Classifier**: A random forest is an ensemble of decision trees. Random forests combine many decision trees in order to reduce the risk of overfitting. Tree ensemble algorithms such as random forests and boosting are among the top performers for classification and regression tasks.
# MAGIC 
# MAGIC We will to compare these models in the AML Workbench _runs_ screen. The next code block creates the model. You can choose between a _DecisionTree_ or _RandomForest_ by setting the 'model_type' variable. We have also included a series of model hyperparameters to guide your exploration of the model space.

# COMMAND ----------

model_type = dbutils.widgets.get("model")

# train a model.
if model_type == 'DecisionTree':
  model = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures",
                                 # Maximum depth of the tree. (>= 0) 
                                 # E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.'
                                 maxDepth=15,
                                 # Max number of bins for discretizing continuous features. 
                                 # Must be >=2 and >= number of categories for any categorical feature.
                                 maxBins=32, 
                                 # Minimum number of instances each child must have after split. 
                                 # If a split causes the left or right child to have fewer than 
                                 # minInstancesPerNode, the split will be discarded as invalid. Should be >= 1.
                                 minInstancesPerNode=1, 
                                 # Minimum information gain for a split to be considered at a tree node.
                                 minInfoGain=0.0, 
                                 # Criterion used for information gain calculation (case-insensitive). 
                                 # Supported options: entropy, gini')
                                 impurity="gini")

  ##=======================================================================================================================
  ## GBTClassifer is only valid for Binary Classifiers, this is a multiclass (failures 1-4) so no GBTClassifier
#elif model_type == 'GBTClassifier':
#  model = GBTClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures",
#                        maxIter=200, stepSize=0.1,
#                        maxDepth=15,
#                        maxBins=32, 
#                        minInstancesPerNode=1, 
#                        minInfoGain=0.0)
  ##=======================================================================================================================
else:
  model = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", 
                                      # Passed to DecisionTreeClassifier
                                      maxDepth=15, 
                                      maxBins=32, 
                                      minInstancesPerNode=1, 
                                      minInfoGain=0.0,
                                      impurity="gini",
                                      # Number of trees to train (>= 1)
                                      numTrees=200, 
                                      # The number of features to consider for splits at each tree node. 
                                      # Supported options: auto, all, onethird, sqrt, log2, (0.0-1.0], [1-n].
                                      featureSubsetStrategy="sqrt", 
                                      # Fraction of the training data used for learning each 
                                      # decision tree, in range (0, 1].' 
                                      subsamplingRate = 0.632)

# chain indexers and model in a Pipeline
pipeline_cls_mthd = Pipeline(stages=[labelIndexer, featureIndexer, model])

# train model.  This also runs the indexers.
model_pipeline = pipeline_cls_mthd.fit(training)

# COMMAND ----------

# MAGIC %md To evaluate this model, we predict the component failures over the test data set. Since the test set has been created from data the model has not been seen before, it simulates future data. The evaluation then can be generalize to how the model could perform when operationalized and used to score the data in real time.

# COMMAND ----------

# Now we need the test data...
test_data = spark.sql("SELECT * FROM " + dbutils.widgets.get("TESTING_TABLE"))
test_data.limit(10).toPandas().head(10)

# assemble features
test_data = va.transform(test_data).select('machineID','dt_truncated','label_e','features')

# set maxCategories so features with > 10 distinct values are treated as continuous.
featureIndexer = VectorIndexer(inputCol="features", 
                               outputCol="indexedFeatures", 
                               maxCategories=10).fit(test_data)

# fit on whole dataset to include all labels in index
labelIndexer = StringIndexer(inputCol="label_e", outputCol="indexedLabel").fit(test_data)

testing = test_data

print(testing.count())

# make predictions. The Pipeline does all the same operations on the test data
predictions = model_pipeline.transform(testing)

# Create the confusion matrix for the multiclass prediction results
# This result assumes a decision boundary of p = 0.5
conf_table = predictions.stat.crosstab('indexedLabel', 'prediction')
confuse = conf_table.toPandas()
confuse.head()

# COMMAND ----------

# MAGIC %md The confusion matrix lists each true component failure in rows and the predicted value in columns. Labels numbered 0.0 corresponds to no component failures. Labels numbered 1.0 through 4.0 correspond to failures in one of the four components in the machine. As an example, the third number in the top row indicates how many days we predicted component 2 would fail, when no components actually did fail. The second number in the second row, indicates how many days we correctly predicted a component 1 failure within the next 7 days.
# MAGIC 
# MAGIC We read the confusion matrix numbers along the diagonal as correctly classifying the component failures. Numbers above the diagonal indicate the model incorrectly predicting a failure when non occured, and those below indicate incorrectly predicting a non-failure for the row indicated component failure.
# MAGIC 
# MAGIC When evaluating classification models, it is convenient to reduce the results in the confusion matrix into a single performance statistic. However, depending on the problem space, it is impossible to always use the same statistic in this evaluation. Below, we calculate four such statistics.
# MAGIC 
# MAGIC - **Accuracy**: reports how often we correctly predicted the labeled data. Unfortunatly, when there is a class imbalance (a large number of one of the labels relative to others), this measure is biased towards the largest class. In this case non-failure days.
# MAGIC 
# MAGIC Because of the class imbalance inherint in predictive maintenance problems, it is better to look at the remaining statistics instead. Here positive predictions indicate a failure.
# MAGIC 
# MAGIC - **Precision**: Precision is a measure of how well the model classifies the truely positive samples. Precision depends on falsely classifying negative days as positive.
# MAGIC 
# MAGIC - **Recall**: Recall is a measure of how well the model can find the positive samples. Recall depends on falsely classifying positive days as negative.
# MAGIC 
# MAGIC - **F1**: F1 considers both the precision and the recall. F1 score is the harmonic average of precision and recall. An F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.
# MAGIC 
# MAGIC These metrics make the most sense for binary classifiers, though they are still useful for comparision in our multiclass setting. Below we calculate these evaluation statistics for the selected classifier, and post them back to the AML workbench run time page for tracking between experiments.

# COMMAND ----------

# select (prediction, true label) and compute test error
# select (prediction, true label) and compute test error
# True positives - diagonal failure terms 
tp = confuse['1.0'][1]+confuse['2.0'][2]+confuse['3.0'][3]+confuse['4.0'][4]

# False positves - All failure terms - True positives
fp = np.sum(np.sum(confuse[['1.0', '2.0','3.0','4.0']])) - tp

# True negatives 
tn = confuse['0.0'][0]

# False negatives total of non-failure column - TN
fn = np.sum(np.sum(confuse[['0.0']])) - tn

# Accuracy is diagonal/total 
acc_n = tn + tp
acc_d = np.sum(np.sum(confuse[['0.0','1.0', '2.0','3.0','4.0']]))
acc = acc_n/acc_d

# Calculate precision and recall.
prec = tp/(tp+fp)
rec = tp/(tp+fn)

# Print the evaluation metrics to the notebook
print("Accuracy = %g" % acc)
print("Precision = %g" % prec)
print("Recall = %g" % rec )
print("F1 = %g" % (2.0 * prec * rec/(prec + rec)))
print("")

# COMMAND ----------

importances = model_pipeline.stages[2].featureImportances

importances

# COMMAND ----------

# MAGIC %md Remember that this is a simulated data set. We would expect a model built on real world data to behave very differently. The accuracy may still be close to one, but the precision and recall numbers would be much lower.
# MAGIC 
# MAGIC ## Persist the model
# MAGIC 
# MAGIC We'll save the latest model for use in deploying a webservice for operationalization in the next notebook. We store this local to the Jupyter notebook kernel because the model is stored in a hierarchical format that does not translate to Azure Blob storage well. 

# COMMAND ----------

# save model
model_pipeline.write().overwrite().save("dbfs:/FileStore/models/" + model_type + ".pqt")
print("Model saved")

# Time the notebook execution. 
# This will only make sense if you "Run All" cells
toc = time.time()
print("Full run took %.2f minutes" % ((toc - tic)/60))

# COMMAND ----------

# MAGIC %sh 
# MAGIC #mkdir /dbfs/FileStore/models
# MAGIC ls -l /dbfs/FileStore/models

# COMMAND ----------

# MAGIC %md ## Conclusion
# MAGIC 
# MAGIC In the next notebook `Code\4_operationalization.ipynb` Jupyter notebook we will create the functions needed to operationalize and deploy any model to get realtime predictions. 