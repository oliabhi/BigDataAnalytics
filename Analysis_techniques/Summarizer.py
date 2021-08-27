from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.ml.stat import Summarizer
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors

master = 'local'
appName = 'ML Summarizer'

config = SparkConf().setAppName(appName).setMaster(master)
sc = SparkContext(conf=config)
# You will need to create the sqlContext
sqlContext = SQLContext(sc)

if sc:
    print(sc.appName)
else:
    print('Could not initialise pyspark session')

df = sc.parallelize([Row(weight=1.0, features=Vectors.dense(1.0, 1.0, 1.0)),
                     Row(weight=0.0, features=Vectors.dense(1.0, 2.0, 3.0))]).toDF()

# create summarizer for multiple metrics "mean" and "count"
summarizer = Summarizer.metrics("mean", "count", "variance", "max", "min")

# compute statistics for multiple metrics with weight
print('=========================')
df1 = df.select(summarizer.summary(df.features, df.weight))
df1.printSchema()
df1.show(truncate=False)

# compute statistics for multiple metrics without weight
print('=========================')
df1 = df.select(summarizer.summary(df.features))
df1.printSchema()
df1.show(truncate=False)

# compute statistics for single metric "mean" with weight
print('=========================')
df1 = df.select(Summarizer.mean(df.features, df.weight))
df1.printSchema()
df1.show(truncate=False)

# compute statistics for single metric "mean" without weight
print('=========================')
df1 = df.select(Summarizer.mean(df.features))
df1.printSchema()
df1.show(truncate=False)

# compute statistics for single metric "count" with weight
print('=========================')
df1 = df.select(Summarizer.count(df.features, df.weight))
df1.printSchema()
df1.show(truncate=False)

# compute statistics for single metric "count" without weight
print('=========================')
df1 = df.select(Summarizer.count(df.features))
df1.printSchema()
df1.show(truncate=False)

# compute statistics for single metric "min" with weight
print('=========================')
df1 = df.select(Summarizer.min(df.features, df.weight))
df1.printSchema()
df1.show(truncate=False)

# compute statistics for single metric "min" without weight
print('=========================')
df1 = df.select(Summarizer.min(df.features))
df1.printSchema()
df1.show(truncate=False)

# compute statistics for single metric "max" with weight
print('=========================')
df1 = df.select(Summarizer.max(df.features, df.weight))
df1.printSchema()
df1.show(truncate=False)

# compute statistics for single metric "max" without weight
print('=========================')
df1 = df.select(Summarizer.max(df.features))
df1.printSchema()
df1.show(truncate=False)
