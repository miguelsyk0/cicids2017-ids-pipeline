import pandas as pd
from data_fetch import fetch_training_data
import time
import os
from dotenv import load_dotenv

# Load environment variables from the .env file in the root folder
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))
def main():
    print("Starting data extraction from Azure SQL (cic_typed)...")
    start_time = time.time()
    
    # Fetch data in chunks to manage memory
    chunks = fetch_training_data(chunksize=100_000, source_table="cic_typed")
    
    # Concatenate all chunks into one massive DataFrame
    print("Fetching and assembling chunks (this may take a few minutes)...")
    full_df = pd.concat(chunks, ignore_index=True)
    
    fetch_time = time.time()
    print(f"Extraction complete! Downloaded {len(full_df):,} rows in {fetch_time - start_time:.2f} seconds.")
    
    # Save to CSV
    print("Saving to CSV format...")
    full_df.to_csv("cic_typed_full.csv", index=False)
    
    # Save to Parquet (Recommended for 2.8M rows - much faster & smaller file size)
    # print("Saving to Parquet format...")
    # full_df.to_parquet("cic_typed_full.parquet", index=False)
    
    print(f"Total time elapsed: {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()