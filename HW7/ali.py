from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("K_mean").getOrCreate()
df = spark.read.csv("C:\\Users\\Asus\\Downloads\\uber-raw-data-aug14.csv", header=True)
from pyspark.ml.feature import StringIndexer
columns_to_index=['Date/Time','Lat','Lon']
for column in columns_to_index:
    indexer=StringIndexer(inputCol=column,outputCol=column+"_N")
    df=indexer.fit(df).transform(df)
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
feat_cols = ['Lat_N', 'Date/Time_N','Lon_N']
assembler = VectorAssembler(inputCols=feat_cols, outputCol='features')
final_df = assembler.transform(df)
from pyspark.ml.feature import StandardScaler
scaler = StandardScaler(inputCol='features', 
                        outputCol='scaled_feat',
                        withStd = True,
                        withMean = False)
scaled_model =scaler.fit(final_df)
cluster_df = scaled_model.transform(final_df)
from pyspark.ml.clustering import KMeans
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
eval = ClusteringEvaluator(predictionCol="prediction",
                           featuresCol="scaled_feat",
                           metricName="silhouette",
                           distanceMeasure="squaredEuclidean")
silhouette_score =[]
print("""
silhoutte scores for K Mean Clustering
======================================
Model\tScore\t
=====\t=====\t
""")
for k in range(2,3):
  kmeans_algo = KMeans(featuresCol='scaled_feat',k=k )
  kmeans_fit = kmeans_algo.fit(cluster_df)
  output = kmeans_fit.transform(cluster_df)
  score = eval.evaluate(output)
  silhouette_score.append(score)
  print(f"K{k}\t{(score)}\t")

train_data, test_data = cluster_df.randomSplit([0.8, 0.2], seed=123)



  # Train KMeans model
kmeans = KMeans(k=k, seed=1)
model = kmeans.fit(train_data)

# Save model
model_path = r"C:/Users/Asus/Desktop/ali/"
model.save(model_path)