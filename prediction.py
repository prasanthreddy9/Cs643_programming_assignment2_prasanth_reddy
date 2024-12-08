# Importing necessary modules
import os
from werkzeug.utils import secure_filename
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from flask import Flask, request, jsonify
from flask_cors import CORS
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

# Creating a Flask app and enabling CORS
app = Flask(__name__)
CORS(app)

# Creating a Spark session named "Prediction"
spark = SparkSession.builder.appName("Prediction").getOrCreate()

# Loading the pre-trained logistic regression model
model = LogisticRegressionModel.load("/src/logistic_regression")

# Defining a route for prediction
@app.route("/predict", methods=["POST"])
def predict():
    # Handling file upload
    file = request.files["file"]
    filename = secure_filename(file.filename)
    os.makedirs("/tmp", exist_ok=True)
    file.save(os.path.join("/tmp", filename))
    
    # Reading and preprocessing validation data
    validation_data = spark.read.schema(schema).options(delimiter=';', header=True, quote='"', ignoreLeadingWhiteSpace=True, ignoreTrailingWhiteSpace=True).csv(os.path.join("/tmp", filename))
    validation_data = validation_data.withColumn("quality", when(col("quality") > 7, 1).otherwise(0))
    feature_columns = validation_data.columns[:-1]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    validation_data = assembler.transform(validation_data)
    
    # Making predictions using the pre-trained model
    predictions = model.transform(validation_data)
    
    # Evaluating predictions using F1 score
    evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")
    f1_score = evaluator.evaluate(predictions)
    
    # Returning the F1 score as JSON
    return jsonify({"f1_score": f1_score})

# Running the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
