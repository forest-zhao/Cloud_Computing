# Category and Trending Correlation
# In order to run this, we use spark-submit, below is the
# spark-submit  \
#   --master local[2] \
    #   category.py
#   --input input-path
#   --output outputfile
##### it seems that I need to improve this by reviewing currently, as this didn't consider framework support/characteristics

from pyspark import SparkContext
from ml_utils import *
import argparse



if __name__ == "__main__":
    sc = SparkContext(appName="category")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="the input path",
                        default='hdfs:///user/forest/')
    parser.add_argument("--output", help="the output path",
                        default='/470374652')
    parser.add_argument("--firstcountry", help="firstcountry",
                        default='GB')
    parser.add_argument("--secondcountry", help="secondcountry",
                        default='US')
    args = parser.parse_args()
    input_path = args.input
    output_path = args.output
    firstcountry=args.firstcountry
    secondcountry = args.secondcountry

    allvideos = sc.textFile(input_path + "allvideos.csv")

    firstCounVideos = allvideos.filter(lambda x:x.split(",")[-1]==firstcountry)
    firstVideoCategory=firstCounVideos.map(extractvideos)
    firstDistinct = firstVideoCategory.distinct()
    firstCategoryTotal=firstDistinct.groupByKey().map(Category)


    secondCounVideos = allvideos.filter(lambda x: x.split(",")[-1] == secondcountry)
    secondVideoCategory = secondCounVideos.map(extractvideos)
    secondDistinct = secondVideoCategory.distinct()

    intersect=firstDistinct.intersection(secondDistinct)
    intersectCategoryTotal=intersect.groupByKey().map(Category)

    final=firstCategoryTotal.join(intersectCategoryTotal).map(categoryTotalPercen)
    finalCollection=final.collect()
    for key,values in finalCollection:
        print(key)
        print(";")
        print("total:")
        print(values[0])
        print("percentage:")
        print(values[1])
        print("in "+secondcountry)
        print("\n")



    final.saveAsTextFile(output_path)
