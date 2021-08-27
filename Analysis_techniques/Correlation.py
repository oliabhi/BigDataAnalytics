from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.sql.functions import col, when

from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import Correlation
import numpy

master = 'local'
appName = 'ML Correlation'

config = SparkConf().setAppName(appName).setMaster(master)
sc = SparkContext(conf=config)
# You will need to create the sqlContext
sqlContext = SQLContext(sc)

if sc:
    print(sc.appName)
else:
    print('Could not initialise pyspark session')

data = [(Vectors.sparse(4, [(0, 1.0), (3, -2.0)]),),
        (Vectors.dense([4.0, 5.0, 0.0, 3.0]),),
        (Vectors.dense([6.0, 7.0, 0.0, 8.0]),),
        (Vectors.sparse(4, [(0, 9.0), (3, 1.0)]),)]

df = sqlContext.createDataFrame(data, ["features"])

df1 = Correlation.corr(df, "features")
df1.printSchema()
r1 = df1.head()
print("Pearson correlation matrix:\n" + str(r1[0]))

df2 = Correlation.corr(df, "features", "spearman")
df2.printSchema()
r2 = df2.head()
print("Spearman correlation matrix:\n" + str(r2[0]))

