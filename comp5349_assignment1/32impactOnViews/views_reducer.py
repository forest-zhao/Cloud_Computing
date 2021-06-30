
#!/usr/bin/python3

import sys
from operator import*


def read_map_output(file):
    """ Return an iterator for key, value pair extracted from file (sys.stdin).
    Input format:  key \t value
    Output format: (key, value)
    """
    for line in file:
        yield line.strip().split("\t", 1)

def views_reducer():

    group_by_key = {}

    inter_array=[]
    for countryvideo,dateviews in read_map_output(sys.stdin):

        # get timestamp and views
        data=dateviews.split(":")
        date=float(data[0])
        views=data[1]


        if countryvideo not in group_by_key:
            group_by_key[(countryvideo)] = []

        group_by_key[(countryvideo)].append((date,views))

    for records in group_by_key.items():

        group_by_key[records[0]].sort()
        if len(group_by_key[records[0]])<2:
            continue

        second=float(group_by_key[records[0]][1][1])
        first=float(group_by_key[records[0]][0][1])
        if first!=0:
           percentage = second/first
           if percentage>=10:

             inter_array.append((records[0],percentage))

   
    final_list={}
    for piece in inter_array:
        x=piece[0].split(":")
        country=x[0]
        vid=x[1]
        fold=piece[1]
        if country not in final_list:
            final_list[country]=[]
        final_list[country].append((fold,vid))
    for final in final_list.items():

        final_list[final[0]].sort(reverse=True)
        for impact in final_list[final[0]]:

            print(final[0]+";",impact[1]+",",impact[0])

if __name__ == "__main__":
    views_reducer()
