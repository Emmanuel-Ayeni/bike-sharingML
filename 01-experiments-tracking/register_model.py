import os
import pickle
import click
import mlflow
import numpy as np
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

HPO_EXPERIMENT_NAME = "sgd-hyperopt-bike"
EXPERIMENT_NAME = "sgd-best-models-bike"

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()

def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

def train_and_log_model(data_path, params):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    # Convert to float32 to reduce memory usage
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)

    with mlflow.start_run():
        params['alpha'] = float(params['alpha'])
        params['max_iter'] = int(params['max_iter'])
        params['eta0'] = float(params['eta0'])

        sgd = SGDRegressor(
            penalty='l2',
            learning_rate='invscaling',
            random_state=42,
            **params
        )

        # Implement batch processing
        batch_size = 10000  # Adjust this based on your available memory
        for _ in range(params['max_iter']):
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                sgd.partial_fit(X_batch, y_batch)

        # Evaluate model on the validation and test sets
        val_rmse = mean_squared_error(y_val, sgd.predict(X_val), squared=False)
        mlflow.log_metric("val_rmse", val_rmse)
        test_rmse = mean_squared_error(y_test, sgd.predict(X_test), squared=False)
        mlflow.log_metric("test_rmse", test_rmse)

@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed bike trip data was saved"
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote"
)
def run_register_model(data_path: str, top_n: int):

    client = MlflowClient()

    # Retrieve the top_n model runs and log the models
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    if experiment is None:
        print(f"Error: Experiment '{HPO_EXPERIMENT_NAME}' not found.")
        print("Please run the hyperparameter optimization script first.")
        return

    # Retrieve the top_n model runs and log the models
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"]
    )
    for run in runs:
        train_and_log_model(data_path=data_path, params=run.data.params)

    # Select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.test_rmse ASC"]
    )[0]

    # Register the best model
    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri, name="sgd-best-model-bike")

if __name__ == '__main__':
    run_register_model()