# gcloud dataproc jobs submit pyspark write_vds.py --cluster cow --files gs://hail-common/hail-hail-is-master-all-spark2.0.2-a205789.jar --py-files gs://hail-common/pyhail-hail-is-master-a205789.zip --properties=spark.driver.extraClassPath=./hail-hail-is-master-all-spark2.0.2-a205789.jar,spark.executor.extraClassPath=./hail-hail-is-master-all-spark2.0.2-a205789.jar

from pyspark import SparkConf, SparkContext
from hail import *

conf = (SparkConf().setAppName("Population Genomics")
		.set("spark.sql.files.openCostInBytes", "1099511627776")
		.set("spark.sql.files.maxPartitionBytes", "1099511627776")
		.set("spark.hadoop.io.compression.codecs", "org.apache.hadoop.io.compress.DefaultCodec,is.hail.io.compress.BGzipCodec,org.apache.hadoop.io.compress.GzipCodec"))

sc = SparkContext(conf=conf)
hc = HailContext(sc)

# Bucket Paths
# src_bucket = "gs://hail-common/"
src_bucket = "gs://genomics-public-data/"
dest_bucket = "gs://bucket_pal/"

# VCF Paths
vcf_path = "sample.vcf"
vcf_path = "/1000-genomes-phase-3/vcf/ALL.chr9.phase3_shapeit2_mvncall_integrated_v2.20130502.genotypes.vcf"

# VDS Paths
#vds_path = "1000Genome/vds/sample.vds"
vds_path = "/1000Genome/vds/phase3_split/ALL.chr9.phase3_shapeit2_mvncall_integrated_v2.20130502.genotypes.vds"

vds = hc.import_vcf(src_bucket + vcf_path).filter_multi().sample_qc()
vds = vds.annotate_samples_table('gs://bucket_pal/1000Genome/sample_info.txt',
						   root='sa.pheno',
						   sample_expr='Sample',
						   config=TextTableConfig(impute=True))

print vds.sample_schema
vds.write(dest_bucket + vds_path)
