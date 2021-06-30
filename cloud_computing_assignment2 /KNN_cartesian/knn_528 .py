# Import all necessary libraries and setup the environment for matplotlib
import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.ml.feature import PCA
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import monotonically_increasing_id
import numpy as np
from pyspark.mllib.evaluation import MulticlassMetrics


train_number=60000
test_number=10000
dimension =50
k_knn=5

def train_rdd(line):
    train_id,labels,pca=line
    return labels,pca

def test_rdd(line):
    pca,test_id=line
    return test_id,pca

def testingset(line):
    test_id,labels,pca=line
    test_labels=sum(labels)
    return test_id, test_labels

def distance(line):
    test_content,train_content=line
    test_id,pca_test=test_content
    train_labels,pca_train=train_content
    error=(pca_test-pca_train).toArray()
    diff= error**2
    sum=np.sum(diff)
    new_distance = sum**0.5
    return test_id,new_distance,train_labels


def change(line):
    test_id,distance,train_labels=line
    return test_id,train_labels

def predict_2(line):
    test_id, list = line
    new_list=list[:k_knn]
    predict=max(new_list,key=new_list.count)
    predict_double=sum(predict)
    return test_id,predict_double

spark = SparkSession \
    .builder \
    .appName("KNN_cartesian final 8") \
    .master("yarn") \
    .config("spark.executor.instances",8) \
    .config("spark.executor.cores",4) \
    .config("spark.executor.memory","4g") \
    .config("spark.default.parallelism",128) \
    .config("spark.sql.shuffle.partition",5000) \
    .getOrCreate()


train_datafile = "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Train-28x28.csv"
train_labelfile = "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Train-label.csv"
num_train_samples = 60000
test_datafile = "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Test-28x28.csv"
test_labelfile= "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Test-label.csv"
num_test_samples = 10000

train_df = spark.read.csv(train_datafile,header=False,inferSchema="true")
test_df = spark.read.csv(test_datafile,header=False,inferSchema="true")
train_la=spark.read.csv(train_labelfile,header=False,inferSchema="true")
test_la=spark.read.csv(test_labelfile,header=False,inferSchema="true")

assembler = VectorAssembler(inputCols=train_df.columns,
    outputCol="features")
train_vectors = assembler.transform(train_df).select("features")

assembler = VectorAssembler(inputCols=test_df.columns,
    outputCol="features")
test_vectors = assembler.transform(test_df).select("features")

assembler = VectorAssembler(inputCols=train_la.columns,
    outputCol="labels")
train_label = assembler.transform(train_la).select("labels")

assembler = VectorAssembler(inputCols=test_la.columns,
    outputCol="labels")
test_label = assembler.transform(test_la).select("labels")



# PCA
pca = PCA(k=dimension, inputCol="features", outputCol="pca")
model= pca.fit(train_vectors)
pca_result_train = model.transform(train_vectors).select('pca')

#model2 = pca.fit(test_vectors)
pca_result_test = model.transform(test_vectors).select('pca')




#   index
train_index =pca_result_train.select("*").withColumn("train_id", monotonically_increasing_id())
test_index=pca_result_test.select("*").withColumn("test_id", monotonically_increasing_id())
tr_label_index=train_label.select("*").withColumn("train_id", monotonically_increasing_id())
ts_label_index=test_label.select("*").withColumn("test_id", monotonically_increasing_id())
#test_index.show(10)

trainset= tr_label_index.join(train_index,"train_id")  #  train_id, labels, pca
testset= ts_label_index.join(test_index,"test_id")     #  test_id, labels, pca

#testset.show(20)


#   rdd
train_rdd=trainset.limit(train_number).rdd.map(train_rdd) # labels, pca
test_rdd=test_index.limit(test_number).rdd.map(test_rdd)  # tes_id,pca

test_original_label= testset.limit(test_number).rdd.map(testingset)    # the original labels in testing set





dis=test_rdd.cartesian(train_rdd).repartition(10000).map(distance)

sort_dis=dis.sortBy(lambda x:x)   #  (test_id,distance,labels) sort by distance





# find the predict labels of each test_id
createCombiner = (lambda x:[x])
mergeVal = (lambda y, x: y+[x])
mergeComb = (lambda y1, y2: y1+y2)
sort_pairrdd=sort_dis.map(change).combineByKey(createCombiner, mergeVal, mergeComb).map(predict_2)  # test_id, predict

#same_labels= test_original_label.intersection(sort_pairrdd)    # the same one

#########
#print('original labels of test set')
#for x in test_original_label.collect():
#      print(x)

#print('predictive labels of test set')
#for y in sort_pairrdd.collect():
 #   print(y)
#############
#    predict,test_labels
def map_predict(line):
    test_id,content=line
    predict,test_labels=content
    predict2=float(predict)
    labels2=float(test_labels)
    return predict2,labels2
test_predict= sort_pairrdd.join(test_original_label).map(map_predict)      # predict, test_labels



metrics = MulticlassMetrics(test_predict)



# Statistics by class
labels = [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]
for label in sorted(labels):
    print("Label %s precision = %s" % (label, metrics.precision(label)))
    print("Label %s recall = %s" % (label, metrics.recall(label)))
    print("Label %s F1 Measure = %s" % (label, metrics.fMeasure(label, beta=1.0)))




spark.stop()





