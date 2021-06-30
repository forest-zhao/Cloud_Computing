import findspark
findspark.init()
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionModel,LogisticRegressionWithLBFGS,LogisticRegressionWithSGD
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler

import numpy as np
from pyspark.mllib.evaluation import MulticlassMetrics


train_number=60000
test_number=10000
iter=100


def change(line):
     label,features=line
     new_label=np.sum(label)
     return new_label,features

def map_c0_pca(line):
    return line._c0,line.features

spark = SparkSession \
    .builder \
    .master("yarn") \
    .config("spark.executor.instances",3) \
    .appName("logistic_regression_change") \
    .config("spark.executor.cores",3) \
    .config("spark.executor.memory","4g") \
    .config("spark.default.parallelism",128) \
    .getOrCreate()

# hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/
train_file = "Train-label-28x28.csv"
test_file = "Test-label-28x28.csv"

test_labelfile= "Test-label.csv"

test_label=np.genfromtxt(test_labelfile,dtype=None)[:test_number]

train_df = spark.read.csv(train_file,header=False,inferSchema="true")

test_df = spark.read.csv(test_file,header=False,inferSchema="true")


assembler = VectorAssembler(inputCols=train_df.columns[1:],outputCol="features")
train_vectors = assembler.transform(train_df).select("_c0","features")

assembler = VectorAssembler(inputCols=test_df.columns[1:],outputCol="features")
test_vectors = assembler.transform(test_df).select("_c0","features")

print("test1")



trainset=train_vectors.limit(train_number).rdd.map(map_c0_pca) #rdd, (train_label,pca)
testset=test_vectors.limit(test_number).rdd.map(map_c0_pca)#rdd(test_lable.pca)


train_rdd_data=trainset.map(change).map(lambda row: LabeledPoint(row[0], [row[1]]))
test_rdd_data=testset.map(change).map(lambda row: LabeledPoint(row[0], [row[1]]))


model=LogisticRegressionWithLBFGS.train(train_rdd_data,numClasses=10,iterations=iter)

print("test2")

# Compute raw scores on the test set
predictionAndLabels = test_rdd_data.map(lambda lp: (float(model.predict(lp.features)), lp.label))

# Instantiate metrics object
metrics = MulticlassMetrics(predictionAndLabels)

# Overall statistics
precision = metrics.precision()

print("Summary Stats")
print("Precision = %s" % precision)


# Statistics by class
labels = test_rdd_data.map(lambda lp: lp.label).distinct().collect()
for label in sorted(labels):
    print("Class %s precision = %s" % (label, metrics.precision(label)))







spark.stop()



