"""
This module includes a few functions used in Category and Trending Correlation
"""
import csv

def extractvideos(record):
    """ This function converts entries of allvideos.csv into key,value pair of the following format
    (category, videoid)

    """
    try:
        csv_list = list(csv.reader([record]))[0]

        return (csv_list[5],csv_list[0])
    except:
        return ()

def Category(record):
    category,videos=record
    return(category,len(videos))


def categoryTotalPercen(record):
    category, values=record
    percentage=(values[1]/values[0])*100
    total=values[0]
    return(category,(total,percentage))


