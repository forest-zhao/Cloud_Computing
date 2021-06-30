# Import all necessary libraries and setup the environment for matplotlib
import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.ml.feature import PCA
from pyspark.ml.feature import VectorAssembler
import numpy as np
from pyspark.mllib.evaluation import MulticlassMetrics


train_number=60000
test_number=10000
dimension = 50
k_knn=5

def train_rdd(line):
    labels,pca=line
    return pca

def label_rdd(line):
    labels,pca=line
    return labels

def test_rdd(line):
    test_id,pca=line
    return test_id,pca

def testingset(line):
    test_id,labels,pca=line
    test_labels=sum(labels)
    return test_id, test_labels

def map_c0_pca(line):
    return line._c0,line.pca

spark = SparkSession \
    .builder \
    .master("yarn") \
    .appName("KNN_numpy") \
    .config("spark.executor.instances",4) \
    .config("spark.executor.cores",4) \
    .config("spark.executor.memory","4g") \
    .config("spark.default.parallelism",128) \
    .config("spark.sql.shuffle.partition",5000) \
    .getOrCreate()


#train_file = "hdfs:///user/weiyuanyuan/Train-label-28x28.csv"
#test_file = "hdfs:///user/weiyuanyuan/Test-label-28x28.csv"
train_file = "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Train-label-28x28.csv"
test_file = "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Test-label-28x28.csv"


train_df = spark.read.csv(train_file,header=False,inferSchema="true")
test_df = spark.read.csv(test_file,header=False,inferSchema="true")


assembler = VectorAssembler(inputCols=train_df.columns[1:],outputCol="features")
train_vectors = assembler.transform(train_df).select("_c0","features")

assembler = VectorAssembler(inputCols=test_df.columns[1:],outputCol="features")
test_vectors = assembler.transform(test_df).select("_c0","features")

#train_vectors.show(20)
#test_vectors.show(20)


########### PCA ##############
pca = PCA(k=dimension, inputCol="features", outputCol="pca")

model= pca.fit(train_vectors)
pca_result_train = model.transform(train_vectors).select('_c0','pca')

#model2= pca.fit(test_vectors)
pca_result_test = model.transform(test_vectors).select('_c0','pca')
#pca_result_train.show(60)
#pca_result_test.show(20)

trainset=pca_result_train.limit(train_number).rdd.map(map_c0_pca)  #rdd, (train_label,pca)
testset=pca_result_test.limit(test_number).rdd.map(map_c0_pca)     #rdd, (test_label,pca)
#for x in trainset.collect():
    #print(x)


#######   numpy   #############
train_rdd=np.asarray(trainset.map(train_rdd).collect())
train_label=np.asarray(trainset.map(label_rdd).collect())
#print(train_label)

############# computer the distance   ############
def knn_distance(line):
    test_label,pca_test=line
    error=(pca_test-train_rdd).toArray()
    diff=error**2
    #b=(pca_test-train_rdd)*(pca_test-train_rdd)
    sum_distance = np.sum(diff,axis=1)
    new_distance = sum_distance**0.5
    return test_label,new_distance

test_rdd=testset.map(knn_distance)  # rdd, (test_label,array(distance))

#for x in test_rdd.collect():
    #print(x)


############ find the k-numbers of labels   ##############
def sort_dis(line):
    multi_train_labels=[]
    test_labels,distance_array,=line
    sort_distance=np.argsort(distance_array)
    k_I=sort_distance[0:k_knn]
    for i in k_I:
        multi_train_labels.append(train_label[i])
    return test_labels,multi_train_labels


test_multi_labels=test_rdd.map(sort_dis)



###########    predict labels #################
def predict_2(line):
    #test_labels, distance_array, result,sort_distance, k_I, new_list = line
    test_labels, new_list = line
    test_labels2=float(test_labels)
    predict=max(new_list,key=new_list.count)
    predict2=float(predict)
    return predict2,test_labels2

test_predict=test_multi_labels.map(predict_2)   # predict,test_labels
predict=test_predict.collect()
#test_predict_array=np.asarray(test_predict.collect())

#print(test_predict_array)

for i in range(10):
    TP = 0.0
    FP = 0.0
    TN = 0.0
    FN = 0.0
    for content in predict:
        if i==content[1]:
            if i==content[0]:
                TP+=1
            else:
                FP+=1
        else:
            if i != content[1]:
                if i == content[0]:
                    TN += 1
                else:
                    FN += 1

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_measure = (2 * TP) / (2 * TP + FP + FN)
    print(i,TP,FP,TN,FN)

    print(i, precision, recall, f1_measure)




spark.stop()





