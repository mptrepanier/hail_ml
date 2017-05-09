# gcloud dataproc jobs submit pyspark ml_on_vds.py --cluster cow --files gs://bucket_pal/hail/build/libs/hail-all-spark.jar --py-files gs://bucket_pal/hail/python/pyhail.zip --properties=spark.driver.extraClassPath=./hail-all-spark.jar,spark.executor.extraClassPath=./hail-all-spark.jar

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, hash
from pyspark.sql.types import *
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, PCA
from hail import *

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

def vds_to_dataframe(vds):

	# vds = quality_control(vds) # Comment out for non-sample.
	vds = vds.filter_samples_expr('sa.pheno.Population == "GBR" || sa.pheno.Population == "PJL" || sa.pheno.Population == "ASW"')
	gkt = vds.genotypes_keytable()
	gdf = gkt.to_dataframe()
	sample_variant_call_df = (gdf.select(gdf['s'].alias('samples'), gdf['`sa.pheno.Population`'].alias('Population'),
										 hash(gdf['`v.contig`'],gdf['`v.start`']).alias('variant_hash'),
										 gdf['`g.gt`'].alias('call')).dropna())

	return sample_variant_call_df


def generate_feature_dataframe(df):

	indexer = StringIndexer(inputCol="variant_hash", outputCol="index")
	indexed_df = indexer.fit(df).transform(df)
	max_index = indexed_df.agg({"index": "max"}).collect()[0].asDict()['max(index)'] + 1


	bmax_index = df.sparkContext.broadcast(max_index) # <------


	vector_rdd = (indexed_df.rdd.map(lambda r: (r[0], (r[3], r[2])))
				  .groupByKey()
				  .mapValues(lambda l: Vectors.sparse((bmax_index + 1), list(l))))
	vector_df = vector_rdd.toDF(['sample', 'features'])
	return vector_df

def get_population_mapping_df(vds):
	skt = vds.samples_keytable()
	sdf = skt.to_dataframe()
	return sdf.select(sdf['s'].alias('samples'), sdf['`sa.pheno.Population`'].alias('Population'))



def build_ml_pipeline():
	pca = PCA(k=10, inputCol="features", outputCol="pcaFeatures")
	kmeans = KMeans(k=3, featuresCol=pca.getOutputCol(), predictionCol="prediction")
	pipeline = Pipeline(stages = [pca, kmeans])
	return pipeline

if __name__=="__main__":

	# Initialize the HailContext.
	hc = HailContext()

	# Read in the Variant DataSet
	src_bucket = "gs://bucket_pal/"

	# vds_path = "1000Genome/vds/sample.vds"
	vds_path = vds_path = "/1000Genome/vds/phase3_split/ALL.chr9.phase3_shapeit2_mvncall_integrated_v2.20130502.genotypes.vds"
	print "\nReading in VDS."
	vds = hc.read(src_bucket + vds_path)

	print "\nMaking Population Mapping DataFrame."
	pm_df = get_population_mapping_df(vds)

	# Build a SparkSession on top of the existing HailContext.
	spark = SparkSession(vds.hc.sc)

	# Convert the VDS to a DataFrame and then construct the feature vectors.
	print "\nGenerating the feature vector DataFrame."
	svc_df = vds_to_dataframe(vds)
	vector_df = generate_feature_dataframe(svc_df)

	# Construct a Spark ML transformation pipeline to act on the data.
	pipeline = build_ml_pipeline()

	# Fit the data to the pipeline and transform it.
	print "\nTraining the model."
	model = pipeline.fit(vector_df)
	result_df = model.transform(vector_df)
	output_df = result_df.join(pm_df, result_df['samples'] == pm_df['samples'])

	model.save(src_bucket + '1000Genome/outputs/fitted_pipeline')
	output_df.write.parquet(src_bucket + '1000Genome/outputs/output.parquet')

	output_df.show()
	spark.stop()


