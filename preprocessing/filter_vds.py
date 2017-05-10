# gcloud dataproc jobs submit pyspark filter_vds.py --cluster cow --files gs://hail-common/hail-hail-is-master-all-spark2.0.2-a205789.jar --py-files gs://hail-common/pyhail-hail-is-master-a205789.zip --properties=spark.driver.extraClassPath=./hail-hail-is-master-all-spark2.0.2-a205789.jar,spark.executor.extraClassPath=./hail-hail-is-master-all-spark2.0.2-a205789.jar

from pyspark.sql import SparkSession
from hail import *

if __name__=="__main__":

	# Initialize the HailContext.
	hc = HailContext()

	# Read in the Variant DataSet
	bucket = "gs://bucket_pal/"

	#vds_path = "1000Genome/vds/sample.vds"
	vds_path = "1000Genome/vds/phase3_split/ALL.chr9.phase3_shapeit2_mvncall_integrated_v2.20130502.genotypes.vds"
	filtered_vds_path = "1000Genome/vds/phase3_split/ALL.chr9.phase3_shapeit2_mvncall_integrated_v2.20130502.genotypes.pop_filtered.vds"
	print "\nReading in VDS."
	vds = hc.read(bucket + vds_path)
	spark = SparkSession(vds.hc.sc)
	filtered_vds = vds.filter_samples_expr(
		'sa.pheno.Population == "GBR" || sa.pheno.Population == "PJL" || sa.pheno.Population == "ASW"')

	filtered_vds.write(bucket + filtered_vds_path)
