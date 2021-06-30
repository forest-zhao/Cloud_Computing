
#!/usr/bin/python3

import sys
import time
import csv



def views_mapper():
    """ This mapper select video_id, country,views and trending_date,
    return the country:video_id,trending_date:views information.

    """

    # to avoid process the first line(title line), set a counter
    i=0

    for line in sys.stdin:

        # read in each line of csv file
        parts=list(csv.reader([line.strip()]))[0]

        # get videoID
        videoID = parts[0].strip()
        # to avoid process the first line(title line)
        if i==0:
           i=1
           continue

        # transform the trending_date into a timestamp
        timeTuple=time.strptime(("20"+parts[1].strip()),'%Y.%d.%m')
        # get trending_date's timestamp
        date=str(time.mktime(timeTuple))
        #get view numbers
        views=parts[8].strip()
        #get country
        country=parts[-1]

        #country:videoID is key, date:views is value
        print("{}\t{}".format(country+":"+videoID, date+":"+views))


if __name__ == "__main__":
    views_mapper()