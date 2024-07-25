import os
import pickle
import click
import mlflow
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.sparse import issparse

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("bike-sharing-ridge")

def load_pickle(filename: str):
    print(f"Loading file from: {filename}")  # Debug print to check the path
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed bike-sharing trip data was saved"
)
@click.option(
    "--alpha",
    default=1.0,
    help="Regularization strength (equivalent to 1/C in SGDRegressor)"
)
@click.option(
    "--max_iter",
    default=1000,
    help="Maximum number of iterations for the solver"
)
def run_train(data_path: str, alpha: float, max_iter: int):
    mlflow.sklearn.autolog()

    X_train_path = os.path.join(data_path, "train.pkl")
    X_val_path = os.path.join(data_path, "val.pkl")

    print(f"Training data path: {X_train_path}")  # Debug print
    print(f"Validation data path: {X_val_path}")  # Debug print

    X_train, y_train = load_pickle(X_train_path)
    X_val, y_val = load_pickle(X_val_path)

    # Convert to float32 to reduce memory usage
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)

    # Convert sparse matrices to dense arrays if necessary
    if issparse(X_train):
        X_train = X_train.toarray()
    if issparse(X_val):
        X_val = X_val.toarray()

    with mlflow.start_run():
        params = {
            "alpha": alpha,
            "max_iter": max_iter,
            "random_state": 42,
            "learning_rate": "invscaling",
            "eta0": 0.01,
            "penalty": "l2"
        }
        sgd = SGDRegressor(**params)
        
        # Implement batch processing
        batch_size = 10000  # Adjust this based on your available memory
        for _ in range(max_iter):
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                sgd.partial_fit(X_batch, y_batch)

        y_pred = sgd.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)
        r2 = r2_score(y_val, y_pred)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        print(f"RMSE: {rmse}")
        print(f"R2 Score: {r2}")

        # Save the model
        mlflow.sklearn.log_model(sgd, "sgd_model")

if __name__ == '__main__':
    run_train()