from utils.spark_session import get_spark
from transformers.cleaning import clean_footpath
import os, shutil

def save_single_csv(df, output_path):
    temp_dir = output_path + "_temp"

    # save part files
    df.coalesce(1).write.csv(temp_dir, header=True, mode="overwrite")

    # find part file
    for f in os.listdir(temp_dir):
        if f.startswith("part-") and f.endswith(".csv"):
            shutil.move(os.path.join(temp_dir, f), output_path)
    
    shutil.rmtree(temp_dir)

def main():
    spark = get_spark("FootpathPhase1")

    # Load RAW data
    df = spark.read.csv("./data/raw/bangkok_traffy.csv", header=True, inferSchema=True)

    # Clean
    df_clean = clean_footpath(df)

    # Save final CSV
    save_single_csv(df_clean, "./data/processed/footpath_phase1.csv")

    print("DONE! Rows:", df_clean.count())

if __name__ == "__main__":
    main()
