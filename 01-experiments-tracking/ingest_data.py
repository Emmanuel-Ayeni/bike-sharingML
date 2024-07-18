import os
import argparse
import pandas as pd
import requests
import urllib.request
from io import BytesIO
from zipfile import ZipFile
import mlflow
import warnings

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

def download_and_extract_data(url: str) -> pd.DataFrame:
    print(f"Downloading data from {url}")
    response = requests.get(url, timeout=30)
    zip_file = ZipFile(BytesIO(response.content))
    
    csv_files = [f for f in zip_file.namelist() if f.endswith('.csv')]
    if not csv_files:
        raise ValueError("No CSV file found in the zip archive")
    
    print(f"Extracting {csv_files[0]}")
    with zip_file.open(csv_files[0]) as file:
        df = pd.read_csv(file)
    
    return df
    

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    print("Available columns in the DataFrame:")
    print(df.columns.tolist())
    
    # Identify the start and end time columns
    start_time_col = next((col for col in df.columns if 'start' in col.lower() and ('time' in col.lower() or 'at' in col.lower())), None)
    end_time_col = next((col for col in df.columns if 'end' in col.lower() and ('time' in col.lower() or 'at' in col.lower())), None)
    
    if not start_time_col or not end_time_col:
        raise ValueError(f"Could not identify start and end time columns. Available columns: {df.columns.tolist()}")
    
    print(f"Using '{start_time_col}' as start time and '{end_time_col}' as end time.")
    
    # Rename columns to match expected format
    df = df.rename(columns={
        start_time_col: 'datetime',
        end_time_col: 'ended_at',
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
    
    # Identify station ID columns
    start_station_col = next((col for col in df.columns if 'start' in col.lower() and 'station' in col.lower() and 'id' in col.lower()), None)
    end_station_col = next((col for col in df.columns if 'end' in col.lower() and 'station' in col.lower() and 'id' in col.lower()), None)
    
    # Identify rideable type column
    rideable_type_col = next((col for col in df.columns if 'rideable' in col.lower() or 'bike' in col.lower()), None)
    
    # Prepare columns to keep
    columns_to_keep = ['datetime', 'year', 'month', 'day', 'hour', 'dayofweek', 'ride_duration', 'user_type']
    if start_station_col:
        columns_to_keep.append(start_station_col)
    if end_station_col:
        columns_to_keep.append(end_station_col)
    if rideable_type_col:
        columns_to_keep.append(rideable_type_col)
    
    # Drop unnecessary columns
    df = df[columns_to_keep]
    
    # Handle categorical variables
    if rideable_type_col:
        df[rideable_type_col] = df[rideable_type_col].astype('category')
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

            print("\nDataFrame info before processing:")
            df.info()

            # Log the raw data shape
            mlflow.log_metric("raw_data_rows", df.shape[0])
            mlflow.log_metric("raw_data_columns", df.shape[1])
            
            # Process the data
            df = process_data(df)
            print("\nDataFrame info after processing:")
            df.info()
            
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
            print("\nDataFrame info:")
            df.info()
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