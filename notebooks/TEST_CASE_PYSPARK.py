# Databricks notebook source
import pyspark.sql.functions as f
from pyspark.sql.window import Window

# COMMAND ----------

df_total_sales = spark.read.option("header",True).csv("/FileStore/mikolaj_halemba/historical_sales_volume.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Yearly rank of products by their sales within each category

# COMMAND ----------

df_grouped_sales = df_total_sales.groupBy("year","product").agg(f.sum("volumeSales").alias("year_sales_per_category"))

window = Window.partitionBy("year").orderBy(f.col("year_sales_per_category").desc())
df_grouped_sales.withColumn("rank", f.rank().over(window)).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## % difference between product sales in a given quarter and median sales of all products in that quarter (regardless of category)

# COMMAND ----------

df_median_sales = df_total_sales.groupBy("year", "quarter").agg(f.percentile_approx("volumeSales", 0.5).alias("median_sales"))
df_total_sales_with_median = df_total_sales.join(df_median_sales, on=["year", "quarter"], how="inner")

df_with_percentage_diff = (df_total_sales_with_median
                           .withColumn("PercentageDifference", f.round((f.col("volumeSales") - f.col("median_sales")) / f.col("median_sales") * 100,2))
)
df_with_percentage_diff.display()

