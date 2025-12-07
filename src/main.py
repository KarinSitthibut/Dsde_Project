from utils.spark_session import get_spark
from pyspark.sql.functions import (
    col, radians, sin, cos, atan2, sqrt, lit, row_number
)
from pyspark.sql.window import Window
import os


# -----------------------------
# Haversine Distance Function
# -----------------------------
def haversine_expr(lat1, lon1, lat2, lon2):
    return (
        lit(6371000.0) * 2 * atan2(
            sqrt(
                sin((radians(lat2 - lat1) / 2))**2 +
                cos(radians(lat1)) * cos(radians(lat2)) *
                sin((radians(lon2 - lon1) / 2))**2
            ),
            sqrt(
                1 - (
                    sin((radians(lat2 - lat1) / 2))**2 +
                    cos(radians(lat1)) * cos(radians(lat2)) *
                    sin((radians(lon2 - lon1) / 2))**2
                )
            )
        )
    )


def main():

    # ------------------------------------------------------------
    # Resolve paths according to YOUR real folder structure
    # ------------------------------------------------------------
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    path_phase1 = os.path.join(project_root, "data", "processed", "footpath_phase1.csv")
    path_district = os.path.join(project_root, "data", "external", "district_offices.csv")

    spark = get_spark("Footpath_AddDistance")


    # ------------------------------------------------------------
    # Load Phase 1 footpath
    # ------------------------------------------------------------
    df_fp = (
        spark.read.csv(path_phase1, header=True)
             .withColumn("lat", col("lat").cast("double"))
             .withColumn("lon", col("lon").cast("double"))
    )

    print("Loaded footpath rows:", df_fp.count())


    # ------------------------------------------------------------
    # Load district office coordinates
    # ------------------------------------------------------------
    df_dist = (
        spark.read.csv(path_district, header=True)
             .withColumn("dist_lat", col("lat").cast("double"))
             .withColumn("dist_lon", col("lon").cast("double"))
    )

    print("Loaded district office rows:", df_dist.count())


    # ------------------------------------------------------------
    # Cross join
    # ------------------------------------------------------------
    df_join = df_fp.crossJoin(df_dist)

    df_join = df_join.withColumn(
        "distance_m",
        haversine_expr(
            col("lat"), col("lon"),
            col("dist_lat"), col("dist_lon")
        )
    )


    # ------------------------------------------------------------
    # For each footpath complaint → pick nearest district office
    # ------------------------------------------------------------
    w = Window.partitionBy("ticket_id").orderBy(col("distance_m").asc())

    df_nearest = (
        df_join.withColumn("rn", row_number().over(w))
               .filter(col("rn") == 1)
               .drop("rn")
    )


    # ------------------------------------------------------------
    # Keep original columns + add distance_m only
    # ------------------------------------------------------------
    original_cols = df_fp.columns  
    df_out = df_nearest.select(*original_cols, "distance_m")


    # ------------------------------------------------------------
    # Save back as same file name (overwrite)
    # ------------------------------------------------------------
    df_out.write.csv(path_phase1, header=True, mode="overwrite")

    print("DONE → Updated footpath_phase1.csv with distance_m")


if __name__ == "__main__":
    main()
