import findspark
findspark.init()
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import PCA
import numpy as np
from pyspark.mllib.evaluation import MulticlassMetrics


train_number=60000
test_number=10000
dimension=784

def change(line):
     label, pca=line
     new_label=np.sum(label)
     new_pca = []
     for a in pca:
         if a >= 0:
             new_pca.append(a)
         else:
             new_pca.append(0)

     return new_label,new_pca

def map_c0_pca(line):
    return line._c0,line.pca

spark = SparkSession \
    .builder \
    .master("yarn") \
    .config("spark.executor.instances",3) \
    .appName("naive bayes,final test 784 3") \
    .config("spark.executor.cores",3) \
    .config("spark.executor.memory","4g") \
    .config("spark.default.parallelism",128) \
    .config("spark.sql.shuffle.partition",5000) \
    .config("spark.shuffle.consolidateFiles","true") \
    .getOrCreate()


train_datafile = "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Train-label-28x28.csv"
test_datafile = "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Test-label-28x28.csv"

#train_datafile = "hdfs:///user/weiyuanyuan/Train-label-28x28.csv"
#test_datafile = "hdfs:///user/weiyuanyuan/Test-label-28x28.csv"

train_df = spark.read.csv(train_datafile,header=False,inferSchema="true")
test_df = spark.read.csv(test_datafile,header=False,inferSchema="true")


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


train_labelpoint=trainset.map(change).map(lambda line: LabeledPoint(line[0], [line[1]]))
test_labelpoint=testset.map(change).map(lambda line: LabeledPoint(line[0], [line[1]]))


model = NaiveBayes.train(train_labelpoint,1.0)


prediction_labels = test_labelpoint.map(lambda x: (float(model.predict(x.features[0])), x.label))

#print("predict labels")
#print(prediction_labels.collect())


metrics = MulticlassMetrics(prediction_labels)

accuracy = metrics.precision()


print("accuracy = %.2f%%" % (accuracy*100))

spark.stop()

