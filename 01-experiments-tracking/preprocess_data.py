import os
import pickle
import click
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)

def read_dataframe(filename: str):
    df = pd.read_csv(filename)
    df['started_at'] = pd.to_datetime(df['started_at'])
    df['ended_at'] = pd.to_datetime(df['ended_at'])
    df['duration'] = (df['ended_at'] - df['started_at']).dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 180)]  # Filter rides between 1 and 180 minutes
    
    categorical = ['start_station_name', 'end_station_name', 'member_casual']
    df[categorical] = df[categorical].astype(str)
    
    return df

def preprocess(df: pd.DataFrame, dv: DictVectorizer, fit_dv: bool = False):
    df['start_hour'] = df['started_at'].dt.hour
    df['start_dayofweek'] = df['started_at'].dt.dayofweek
    df['start_month'] = df['started_at'].dt.month
    
    categorical = ['start_station_name', 'end_station_name', 'member_casual']
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
    help="Location where the raw Capital Bikeshare data is saved"
)
@click.option(
    "--dest_path",
    help="Location where the resulting files will be saved"
)
def run_data_prep(raw_data_path: str, dest_path: str):
    # Load CSV files
    df_2023_01 = read_dataframe(os.path.join(raw_data_path, "202301-capitalbikeshare-tripdata.csv"))
    df_2023_02 = read_dataframe(os.path.join(raw_data_path, "202302-capitalbikeshare-tripdata.csv"))
    df_2023_03 = read_dataframe(os.path.join(raw_data_path, "202303-capitalbikeshare-tripdata.csv"))
    
    # Combine dataframes
    df = pd.concat([df_2023_01, df_2023_02, df_2023_03])
    
    # Split the data
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    df_train, df_val = train_test_split(df_train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2
    
    # Extract the target
    target = 'duration'
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