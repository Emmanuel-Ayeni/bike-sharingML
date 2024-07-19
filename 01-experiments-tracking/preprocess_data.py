import os
import pickle
import click
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        pickle.dump(obj, f_out)

def read_dataframe(filename: str):
    df = pd.read_parquet(filename)
    print(f"Reading file: {filename}")
    print(f"Columns in the DataFrame: {df.columns.tolist()}")
    print(f"First few rows of the DataFrame:")
    print(df.head())

    # Check if the required columns exist, with possible alternative names
    column_mapping = {
        'started_at': ['started_at', 'start_time', 'start_date', 'datetime'],
        'ended_at': ['ended_at', 'end_time', 'end_date']
    }

    for required_col, alternatives in column_mapping.items():
        if required_col not in df.columns:
            for alt_col in alternatives:
                if alt_col in df.columns:
                    df[required_col] = df[alt_col]
                    print(f"Using alternative column '{alt_col}' for '{required_col}'")
                    break
            else:
                available_columns = df.columns.tolist()
                raise ValueError(f"Missing required column '{required_col}' and no suitable alternative found in {filename}. Available columns: {available_columns}")

    # If columns exist, proceed with duration calculation
    df['ride_duration'] = (pd.to_datetime(df['ended_at']) - pd.to_datetime(df['started_at'])).dt.total_seconds() / 60
    df = df[(df.ride_duration >= 1) & (df.ride_duration <= 180)]  # Filter rides between 1 and 180 minutes
    
    categorical = ['start_station_name', 'end_station_name', 'user_type']
    for col in categorical:
        if col in df.columns:
            df[col] = df[col].astype('category')
            df[col] = df[col].cat.add_categories(['Unknown'])
            df[col] = df[col].fillna('Unknown')
    
    return df

def preprocess(df: pd.DataFrame, dv: DictVectorizer, fit_dv: bool = False):
    df['started_at'] = pd.to_datetime(df['started_at'])  # Ensure started_at is datetime
    df['start_hour'] = df['started_at'].dt.hour
    df['start_dayofweek'] = df['started_at'].dt.dayofweek
    df['start_month'] = df['started_at'].dt.month
    
    categorical = ['start_station_name', 'end_station_name', 'user_type']
    numerical = ['start_hour', 'start_dayofweek', 'start_month']
    
    dicts = df[categorical + numerical].to_dict(orient='records')
    
    if fit_dv:
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    
    return X, dv

@click.command()
@click.option(
    "--raw_data_path",
    help="Location where the raw Capital Bikeshare data in Parquet format is saved"
)
@click.option(
    "--dest_path",
    help="Location where the resulting files will be saved" 
)

def run_data_prep(raw_data_path: str, dest_path: str):
    # Load Parquet files
    df_2022_01 = read_dataframe(os.path.join(raw_data_path, "bike_data_2022_01.parquet"))
    df_2022_02 = read_dataframe(os.path.join(raw_data_path, "bike_data_2022_02.parquet"))
    df_2022_03 = read_dataframe(os.path.join(raw_data_path, "bike_data_2022_03.parquet"))
    
    # Combine dataframes
    df = pd.concat([df_2022_01, df_2022_02, df_2022_03])
    
    # Split the data
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    df_train, df_val = train_test_split(df_train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2
    
    # Extract the target
    target = 'ride_duration'
    y_train = df_train[target].values
    y_val = df_val[target].values
    y_test = df_test[target].values
    
    # Fit the DictVectorizer and preprocess data
    dv = DictVectorizer()
    X_train, dv = preprocess(df_train, dv, fit_dv=True)
    X_val, _ = preprocess(df_val, dv, fit_dv=False)
    X_test, _ = preprocess(df_test, dv, fit_dv=False)
    
    # Create dest_path folder unless it already exists
    os.makedirs(dest_path, exist_ok=True)
    
    # Save DictVectorizer and datasets
    dump_pickle(dv, os.path.join(dest_path, "dv.pkl"))
    dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    dump_pickle((X_val, y_val), os.path.join(dest_path, "val.pkl"))
    dump_pickle((X_test, y_test), os.path.join(dest_path, "test.pkl"))

if __name__ == '__main__':
    run_data_prep()
