from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import ChiSquareTest

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


data = [(0.0, Vectors.dense(0.5, 10.0)),
        (0.0, Vectors.dense(1.5, 20.0)),
        (1.0, Vectors.dense(1.5, 30.0)),
        (0.0, Vectors.dense(3.5, 30.0)),
        (0.0, Vectors.dense(3.5, 40.0)),
        (1.0, Vectors.dense(3.5, 40.0))]
df = sqlContext.createDataFrame(data, ["label", "features"])
df.printSchema()

df1 = ChiSquareTest.test(df, "features", "label")
df1.printSchema()
r = df1.head()

print("pValues: " + str(r.pValues))
print("degreesOfFreedom: " + str(r.degreesOfFreedom))
print("statistics: " + str(r.statistics))
