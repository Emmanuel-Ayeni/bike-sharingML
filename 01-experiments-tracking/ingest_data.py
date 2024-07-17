import os
import argparse
import pandas as pd
import urllib.request
from io import BytesIO
from zipfile import ZipFile
import mlflow
import warnings

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

"""def download_and_extract_data(url: str) -> pd.DataFrame:
    print(f"Downloading data from {url}")
    response = requests.get(url, timeout=30)
    zip_file = ZipFile(BytesIO(response.content))
    
    csv_files = [f for f in zip_file.namelist() if f.endswith('.csv')]
    if not csv_files:
        raise ValueError("No CSV file found in the zip archive")
    
    print(f"Extracting {csv_files[0]}")
    with zip_file.open(csv_files[0]) as file:
        df = pd.read_csv(file)
    
    return df"""

import urllib.request

def download_and_extract_data(url: str) -> pd.DataFrame:
    print(f"Downloading data from {url}")
    file_name, _ = urllib.request.urlretrieve(url)
    zip_file = ZipFile(file_name)
    
    csv_files = [f for f in zip_file.namelist() if f.endswith('.csv')]
    if not csv_files:
        raise ValueError("No CSV file found in the zip archive")
    
    print(f"Extracting {csv_files[0]}")
    with zip_file.open(csv_files[0]) as file:
        df = pd.read_csv(file)
    
    return df

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    # Rename columns to match expected format
    df = df.rename(columns={
        'started_at': 'datetime',
        'member_casual': 'user_type'
    })
    
    # Convert datetime and extract components
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    
    # Calculate ride duration in minutes
    df['ride_duration'] = (pd.to_datetime(df['ended_at']) - df['datetime']).dt.total_seconds() / 60
    
    # Drop unnecessary columns
    columns_to_keep = ['datetime', 'year', 'month', 'day', 'hour', 'dayofweek',
                       'ride_duration', 'start_station_id', 'end_station_id', 'rideable_type', 'user_type']
    df = df[columns_to_keep]
    
    # Handle categorical variables
    df['rideable_type'] = df['rideable_type'].astype('category')
    df['user_type'] = df['user_type'].astype('category')
    
    return df

def ingest_data(year: int, month: int, output_path: str):
    print(f"Starting data ingestion for {year}-{month}")
    mlflow.set_experiment("bike-sharing-data-ingestion")
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("year", year)
        mlflow.log_param("month", month)
        mlflow.log_param("output_path", output_path)
        
        # Construct the URL using the year and month
        url = f"https://s3.amazonaws.com/capitalbikeshare-data/{year:04d}{month:02d}-capitalbikeshare-tripdata.zip"
        
        try:
            # Download and extract data
            df = download_and_extract_data(url)
            
            # Log the raw data shape
            mlflow.log_metric("raw_data_rows", df.shape[0])
            mlflow.log_metric("raw_data_columns", df.shape[1])
            
            # Process the data
            df = process_data(df)
            
            # Log the processed data shape
            mlflow.log_metric("processed_data_rows", df.shape[0])
            mlflow.log_metric("processed_data_columns", df.shape[1])
            
            # Ensure the output directory exists
            os.makedirs(output_path, exist_ok=True)
            
            # Save the processed dataframe
            output_file = os.path.join(output_path, f'bike_data_{year:04d}_{month:02d}.parquet')
            df.to_parquet(output_file, index=False)
            
            # Log the output file as an artifact
            mlflow.log_artifact(output_file)
            
            print(f"Ingested data saved to {output_file}")
            print(f"Shape of the ingested data: {df.shape}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            raise
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ingest bike sharing dataset")
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Year of the data to ingest (YYYY)"
    )
    parser.add_argument(
        "--month",
        type=int,
        required=True,
        help="Month of the data to ingest (1-12)"
    )
    parser.add_argument(
        "--output_path",
        default="./output",
        help="Path to save the processed data"
    )
    
    args = parser.parse_args()
    
    ingest_data(args.year, args.month, args.output_path)