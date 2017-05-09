# gcloud dataproc jobs submit pyspark ml_on_vds.py --cluster cow --files gs://bucket_pal/hail/build/libs/hail-all-spark.jar --py-files gs://bucket_pal/hail/python/pyhail.zip --properties=spark.driver.extraClassPath=./hail-all-spark.jar,spark.executor.extraClassPath=./hail-all-spark.jar

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, hash
from pyspark.sql.types import *
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StringIndexer
from hail.expr import *

def quality_control(vds):
	# Filter on allelic balance.
	filter_condition_ab = '''let ab = g.ad[1] / g.ad.sum() in
	                         ((g.isHomRef() && ab <= 0.1) ||
	                          (g.isHet() && ab >= 0.25 && ab <= 0.75) ||
	                          (g.isHomVar() && ab >= 0.9))'''

	# Filter out variants with low call rates.
	vds_gAB = vds.filter_genotypes(filter_condition_ab)
	vds_gAB_vCR = (vds_gAB
				   .filter_variants_expr('gs.fraction(g => g.isCalled()) > 0.95')
				   .sample_qc())

	return vds_gAB_vCR

# Initialize the first SparkSession
spark1 = (SparkSession.builder.appName("HailContext")
		.config("spark.sql.files.openCostInBytes", "1099511627776")
		.config("spark.sql.files.maxPartitionBytes", "1099511627776")
		.config("spark.hadoop.io.compression.codecs", "org.apache.hadoop.io.compress.DefaultCodec,is.hail.io.compress.BGzipCodec,org.apache.hadoop.io.compress.GzipCodec")
		.getOrCreate())

hc = HailContext(spark1.sparkContext)

# Read in the Variant DataSet
src_bucket = "gs://bucket_pal/"
vds_path = "1000Genome/vds/sample.vds"
vds = hc.read(src_bucket + vds_path)

# Perform Quality Control on it.
vds = quality_control(vds)
gkt = vds.genotypes_keytable()
gdf = gkt.to_dataframe()
gdf.write.mode("overwrite").parquet(src_bucket + "1000Genome/temp/gdf.parquet")
hc.stop()
spark1.stop()


# Reinitialize Spark without Hail.
spark2 = SparkSession.builder.appName("PopulationGenomics").getOrCreate()
gdf = spark2.read.parquet(src_bucket + "1000Genome/temp/gdf.parquet")
svc_df = (gdf.select(gdf['s'], hash(gdf['`v.contig`'], gdf['`v.start`']).alias('variant_hash'), gdf['`g.gt`'].alias('call'))
		  .dropna())

indexer = StringIndexer(inputCol="variant_hash", outputCol="index")
indexed_df = indexer.fit(svc_df).transform(svc_df)
max_index = indexed_df.agg({"index": "max"}).collect()[0].asDict()['max(index)']

vector_rdd = (indexed_df.rdd.map(lambda r: (r[0], (r[3], r[2])))
			  .groupByKey()
			  .mapValues(lambda l: Vectors.sparse((max_index + 1), list(l))))

vector_df = vector_rdd.toDF(['s', 'features'])
vector_df.show()
pyspark.sql
kmeans = KMeans().setK(3)
model = kmeans.fit(vector_df)
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
	print center

spark2.stop()


