#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Invalid number of parameters!"
    echo "Usage: ./views_driver.sh [input_location] [output_location]"
    exit 1
fi

hadoop jar $HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-2.9.0.jar \
-D mapreduce.job.reduces=3 \
-D mapreduce.job.name='impact of Trending on View Number' \
-file views_mapper.py \
-mapper 'python views_mapper.py' \
-file views_reducer.py \
-reducer 'python views_reducer.py' \
-input $1 \
-output $2
