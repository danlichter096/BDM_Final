from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
import functools
import datetime
import json
import numpy as np
import sys


def expandVisits(date_range_start, visits_by_day):
    visits = []
    visits_by_day = list(map(lambda x: int(x), visits_by_day.replace('[','').replace(']', '').split(',')))
    truncated_date = date_range_start[0:10]
    datetime_date = datetime.datetime.strptime(truncated_date, '%Y-%m-%d')
    for x in visits_by_day:
        year = int(datetime_date.isoformat()[0:4])
        date = datetime_date.isoformat()[5:10]
        datetime_date+=datetime.timedelta(days=1)
        visits.append([year,date,x])
    return visits

def computeStats(groupCount, group, visits):
    counts = groupCount[group]
    visits = np.array(visits)
    visits.resize(counts)
    median = int(np.ceil(np.median(visits)))
    std = int(np.round(np.std(visits)))
    high = median+std 
    if(median-std<0):
      low = 0
    else:
      low = median-std
    return (median, low, high)

def main(sc, spark):
    '''
    Transfer our code from the notebook here, however, remember to replace
    the file paths with the ones provided in the problem description.
    '''
    dfPlaces = spark.read.csv('/data/share/bdm/core-places-nyc.csv', header=True, escape='"')
    dfPattern = spark.read.csv('/data/share/bdm/weekly-patterns-nyc-2019-2020/*', header=True, escape='"')
    OUTPUT_PREFIX = sys.argv[1]
    CAT_CODES = set(['452210', '452311', '445120', '722410', '722511', '722513', '446110', '446191','311811', '722515', 
             '445210','445220','445230','445291','445292','445299','445110'])
    CAT_GROUP = {'452311': 0, '452210': 0, '445120': 1, '722410': 2, '722511': 3, '722513': 4, '446191': 5, 
             '446110': 5, '722515': 6, '311811': 6, '445299': 7, '445220': 7, '445292': 7, '445291': 7, '445230': 7, '445210': 7, '445110': 8}
    
    udfToGroup = F.udf(CAT_GROUP.get, T.IntegerType())
    dfWantedPlaces = dfPlaces.select('placekey','naics_code')\
                             .where(F.col('naics_code').isin(CAT_CODES))\
                             .withColumn('group', udfToGroup('naics_code'))\
                             .drop('naics_code').cache()

    groupCount = dict(dfWantedPlaces.groupBy('group').count().collect())
    
    visitType = T.StructType([T.StructField('year', T.IntegerType()),
                              T.StructField('date', T.StringType()),
                              T.StructField('visits', T.IntegerType())])

    statsType = T.StructType([T.StructField('median', T.IntegerType()),
                              T.StructField('low', T.IntegerType()),
                              T.StructField('high', T.IntegerType())])
    udfExpand = F.udf(expandVisits, T.ArrayType(visitType))        
    udfComputeStats = F.udf(functools.partial(computeStats, groupCount), statsType)    

    dfH = dfPattern.join(dfWantedPlaces, 'placekey') \
                   .withColumn('expanded', F.explode(udfExpand('date_range_start', 'visits_by_day'))) \
                   .select('group', 'expanded.*')\
                   .where(F.col('year')>2018)
    dfH.write.csv(f'{OUTPUT_PREFIX}/test',mode='overwrite', header=True)
    #dfI = dfH.groupBy('group', 'year', 'date') \
    #        .agg(F.collect_list('visits').alias('visits')) \
    #        .withColumn('stats', udfComputeStats('group', 'visits')).select('stats.*')
   
    #dfJ = dfI \
    #    .select('group','year','date','stats.*').orderBy('group','year','date')\
    #    .withColumn('date',F.concat(F.lit('2020-'),F.col('date')))\
    #    .cache()
    #dfI.write.csv(f'{OUTPUT_PREFIX}/test',mode='overwrite', header=True)
    

if __name__=='__main__':
    sc = SparkContext()
    spark = SparkSession(sc)
    main(sc, spark)
    
        
    
