# gcloud dataproc jobs submit pyspark ml_on_vds.py --cluster cow --files gs://bucket_pal/hail/build/libs/hail-all-spark.jar --py-files gs://bucket_pal/hail/python/pyhail.zip --properties=spark.driver.extraClassPath=./hail-all-spark.jar,spark.executor.extraClassPath=./hail-all-spark.jar

from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.sql.functions import udf, hash, broadcast
from pyspark.sql.types import *
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, PCA
from hail import *


def quality_control(vds):
    """
    Filters out alleles which are in linkage disequillibrium,
    as well as those with an allelic frequency < 0.4

    input: vds
    output: dataframe
    """
    qc_vds = (vds.ld_prune(memory_per_core=13312, num_cores=8)
              .variant_qc()
              .filter_variants_expr("va.qc.AF > 0.4"))

    qc_vds = qc_vds.repartition(96, False)
    return qc_vds


def vds_to_dataframe(vds):
    """
    Performs quality control on the input vds,
    converts it to a Genomics Keytable, and converts that to a
    sample/variant/call dataframe.

    input: vds
    output: dataframe
    """

    qc_vds = quality_control(vds)
    gkt = qc_vds.genotypes_keytable()
    gdf = gkt.to_dataframe()
    sample_variant_call_df = (gdf.select(gdf['s'].alias('samples'),
                                         hash(gdf['`v.contig`'], gdf['`v.start`']).alias('variant_hash'),
                                         gdf['`g.gt`'].alias('call')).dropna())

    return sample_variant_call_df


def generate_feature_dataframe(df, spark):
    """
    Takes in a sample_variant_call() dataframe
    and maps it to feature vectors which can be used in an ML setting.

    input: dataframe, SparkSession
    output: dataframe
    """
    indexer = StringIndexer(inputCol="variant_hash", outputCol="index")
    indexed_df = indexer.fit(df).transform(df)
    max_index = indexed_df.agg({"index": "max"}).collect()[0].asDict()['max(index)'] + 1
    bmax_index = spark.sparkContext.broadcast(max_index + 1)
    vector_rdd = (indexed_df.rdd.map(lambda r: (r[0], (r[3], r[2])))
                  .groupByKey()
                  .mapValues(lambda l: Vectors.sparse((bmax_index.value), list(l))))

    vector_df = vector_rdd.toDF(['samples', 'features'])
    return vector_df


def get_population_mapping_df(vds):
    """
    Constructs a dataframe which is a mapping of samples to their
    respective populations.

    input: vds
    output: dataframe
    """
    vds = vds.filter_samples_expr(
        'sa.pheno.Population == "GBR" || sa.pheno.Population == "PJL" || sa.pheno.Population == "ASW"')
    skt = vds.samples_keytable()
    sdf = skt.to_dataframe()
    return sdf.select(sdf['s'].alias('samples'), sdf['`sa.pheno.Population`'].alias('Population'))


def build_ml_pipeline():
    """
    Builds a Spark machine learning pipeline for PCA/K-Means.
    """
    pca = PCA(k=30, inputCol="features", outputCol="pcaFeatures")
    kmeans = KMeans(k=3, featuresCol=pca.getOutputCol(), predictionCol="prediction")
    pipeline = Pipeline(stages=[pca, kmeans])
    return pipeline


if __name__ == "__main__":
    # Initialize the SparkContext/HailContext.
    conf = (SparkConf().setAppName("Population Genomics")
            .set("spark.sql.files.openCostInBytes", "1099511627776")
            .set("spark.sql.files.maxPartitionBytes", "1099511627776")
            .set("spark.kryoserializer.buffer.max", "1024")
            .set("spark.hadoop.io.compression.codecs",
                 "org.apache.hadoop.io.compress.DefaultCodec,is.hail.io.compress.BGzipCodec,org.apache.hadoop.io.compress.GzipCodec"))

    sc = SparkContext(conf=conf)
    hc = HailContext(sc)

    # Read in the Variant DataSet
    src_bucket = "gs://bucket_pal/"

    # vds_path = "1000Genome/vds/sample.vds"
    vds_path = vds_path = "/1000Genome/vds/phase3_split/ALL.chr9.phase3_shapeit2_mvncall_integrated_v2.20130502.genotypes.pop_filtered.vds"

    print "\nReading in VDS."
    vds = hc.read(src_bucket + vds_path)

    # Build a SparkSession on top of the existing HailContext.
    spark = SparkSession(vds.hc.sc)

    # Convert the VDS to a DataFrame and then construct the feature vectors.
    print "\nConverting to DataFrame."
    svc_df = vds_to_dataframe(vds)

    print "\nGenerating the feature vectors."
    vector_df = generate_feature_dataframe(svc_df, spark)

    # Construct a Spark ML transformation pipeline to act on the data.
    pipeline = build_ml_pipeline()

    # Fit the data to the pipeline and transform it.
    print "\nTraining the model."
    model = pipeline.fit(vector_df)
    result_df = model.transform(vector_df)

    print "\nMaking Population Mapping DataFrame."
    pm_df = get_population_mapping_df(vds)

    print "\nJoining the DataFrames."
    output_df = result_df.join(broadcast(pm_df), ['samples'])
    output_df.show()
    model.save(src_bucket + '1000Genome/outputs/fitted_pipeline')
    output_df.write.parquet(src_bucket + '1000Genome/outputs/output.parquet')
    reduced_df = output_df.select(output_df['samples'], output_df['prediction'], output_df['Population']).coalesce(1)
    reduced_df.write.csv(src_bucket + '1000Genome/outputs/reduced.csv')

    spark.stop()
