# Importing necessary modules from PySpark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col, when

# Defining the schema for the dataset
schema = StructType([
    StructField("fixed acidity", DoubleType(), True),
    StructField("volatile acidity", DoubleType(), True),
    StructField("citric acid", DoubleType(), True),
    StructField("residual sugar", DoubleType(), True),
    StructField("chlorides", DoubleType(), True),
    StructField("free sulfur dioxide", DoubleType(), True),
    StructField("total sulfur dioxide", DoubleType(), True),
    StructField("density", DoubleType(), True),
    StructField("pH", DoubleType(), True),
    StructField("sulphates", DoubleType(), True),
    StructField("alcohol", DoubleType(), True),
    StructField("quality", DoubleType(), True)
])

# Creating a Spark session named "Training"
spark = SparkSession.builder.appName("Training").getOrCreate()

# Reading the training dataset with specified schema and options
training_dataset = spark.read.schema(schema).options(delimiter=';', header=True, quote='"', ignoreLeadingWhiteSpace=True, ignoreTrailingWhiteSpace=True).csv('file:///home/ec2-user/TrainingDataset.csv')

# Removing double quotes from column names
training_dataset = training_dataset.toDF(*[col.replace('"', '') for col in training_dataset.columns])

# Converting the "quality" column to binary values (1 if greater than 7, else 0)
training_dataset = training_dataset.withColumn("quality", when(col("quality") > 7, 1).otherwise(0))

# Extracting feature columns and assembling them into a vector
feature_columns = training_dataset.columns[:-1]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
training_dataset = assembler.transform(training_dataset)

# Splitting the dataset into training and testing sets
(training_set, testing_set) = training_dataset.randomSplit([0.8, 0.2])

# Creating a logistic regression model
logistic_regression = LogisticRegression(labelCol="quality", featuresCol="features")

# Training the model on the training set
model = logistic_regression.fit(training_set)

# Making predictions on the testing set
predictions = model.transform(testing_set)

# Evaluating the model using the F1 score
multiclass_evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")
f1 = multiclass_evaluator.evaluate(predictions)

# Printing the F1 score
print("F1 Score: {:.4f}".format(f1))

# Saving the trained logistic regression model
model.save("file:///home/ec2-user/Cs643_programming_assignment2_prasanth_reddy/logistic_regression")
