import findspark
findspark.init()
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionModel,LogisticRegressionWithLBFGS,LogisticRegressionWithSGD
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import PCA
import numpy as np
from pyspark.mllib.evaluation import MulticlassMetrics

dimension=128
train_number=60000
test_number=10000
iter=100


def change(line):
     label,pca=line
     new_label=np.sum(label)
     return new_label,pca

def map_c0_pca(line):
    return line._c0,line.pca

spark = SparkSession \
    .builder \
    .master("yarn") \
    .config("spark.executor.instances",3) \
    .appName("logistic_regression_reduced") \
    .config("spark.executor.cores",3) \
    .config("spark.executor.memory","4g") \
    .config("spark.default.parallelism",128) \
    .config("spark.sql.shuffle.partition",5000) \
    .config("spark.shuffle.consolidateFiles","true") \
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


# PCA
pca = PCA(k=dimension, inputCol="features", outputCol="pca")
model = pca.fit(train_vectors)
pca_result_train = model.transform(train_vectors).select('_c0','pca')

#model2 = pca.fit(test_vectors)
pca_result_test = model.transform(test_vectors).select('_c0','pca')
#pca_result_train.show(20)
#pca_result_test.show(20)

trainset=pca_result_train.limit(train_number).rdd.map(map_c0_pca) #rdd, (train_label,pca)
testset=pca_result_test.limit(test_number).rdd.map(map_c0_pca)#rdd(test_lable.pca)


train_rdd_data=trainset.map(change).map(lambda row: LabeledPoint(row[0], [row[1]]))
test_rdd_data=testset.map(change).map(lambda row: LabeledPoint(row[0], [row[1]]))


model=LogisticRegressionWithLBFGS.train(train_rdd_data,numClasses=10,iterations=iter)



# Compute raw scores on the test set
predictionAndLabels = test_rdd_data.map(lambda lp: (float(model.predict(lp.features)), lp.label))

# Instantiate metrics object
metrics = MulticlassMetrics(predictionAndLabels)

# Overall statistics
precision = metrics.precision()
#recall = metrics.recall()
#f1Score = metrics.fMeasure()
print("Summary Stats")
print("Precision = %s" % precision)
#print("Recall = %s" % recall)
#print("F1 Score = %s" % f1Score)

# Statistics by class
#labels = test_rdd_data.map(lambda lp: lp.label).distinct().collect()
#for label in sorted(labels):
   # print("Class %s precision = %s" % (label, metrics.precision(label)))
   # print("Class %s recall = %s" % (label, metrics.recall(label)))
   # print("Class %s F1 Measure = %s" % (label, metrics.fMeasure(label, beta=1.0)))

# Weighted stats
#print("Weighted recall = %s" % metrics.weightedRecall)
#print("Weighted precision = %s" % metrics.weightedPrecision)
#print("Weighted F(1) Score = %s" % metrics.weightedFMeasure())
#print("Weighted F(0.5) Score = %s" % metrics.weightedFMeasure(beta=0.5))
#print("Weighted false positive rate = %s" % metrics.weightedFalsePositiveRate)
spark.stop()








