from pyspark.sql.functions import (
    col, split, to_timestamp, unix_timestamp, lit, lower, coalesce, expr, percentile_approx
)

def clean_footpath(df):
    # 1) Keyword filter
    df_fp = df.filter(
        lower(coalesce(col("type").cast("string"), lit(""))).contains("ทางเท้า") |
        lower(coalesce(col("type").cast("string"), lit(""))).contains("ฟุตบาท") |
        lower(coalesce(col("type").cast("string"), lit(""))).contains("เดิน") |
        lower(coalesce(col("comment").cast("string"), lit(""))).contains("ทางเท้า") |
        lower(coalesce(col("comment").cast("string"), lit(""))).contains("ฟุตบาท") |
        lower(coalesce(col("comment").cast("string"), lit(""))).contains("เดิน")
    )

    # 2) Split coords
    df_fp = df_fp.withColumn("lon", split(col("coords").cast("string"), ",")[0].cast("double"))
    df_fp = df_fp.withColumn("lat", split(col("coords").cast("string"), ",")[1].cast("double"))

    # 3) Convert timestamps
    df_fp = df_fp.withColumn("timestamp", to_timestamp("timestamp"))
    df_fp = df_fp.withColumn("last_activity", to_timestamp("last_activity"))

    # 4) Duration hours
    df_fp = df_fp.withColumn(
        "duration_hours",
        (unix_timestamp("last_activity") - unix_timestamp("timestamp")) / 3600
    )

    # 5) Filter duration > 0
    df_fp = df_fp.filter(col("duration_hours") > 0)

    # 6) Keep “เสร็จสิ้น”
    df_fp = df_fp.filter(col("state") == "เสร็จสิ้น")

    # 7) Percentile clipping
    bounds = df_fp.agg(
        percentile_approx("duration_hours", 0.01).alias("lower"),
        percentile_approx("duration_hours", 0.99).alias("upper")
    ).collect()[0]

    lower_bound = float(bounds["lower"])
    upper_bound = float(bounds["upper"])

    df_fp = df_fp.withColumn(
        "duration_hours",
        expr(f"CASE WHEN duration_hours < {lower_bound} THEN {lower_bound} "
             f"WHEN duration_hours > {upper_bound} THEN {upper_bound} "
             f"ELSE duration_hours END")
    )

    return df_fp