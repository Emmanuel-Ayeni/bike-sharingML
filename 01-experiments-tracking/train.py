import os
import pickle
import click
import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("bike-sharing-random-forest")

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed bike-sharing trip data was saved"
)
@click.option(
    "--num_trees",
    default=100,
    help="Number of trees in the Random Forest"
)
@click.option(
    "--max_depth",
    default=10,
    help="Max depth of the trees in the Random Forest"
)
def run_train(data_path: str, num_trees: int, max_depth: int):
    mlflow.sklearn.autolog()

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    with mlflow.start_run():
        params = {"n_estimators": num_trees, "max_depth": max_depth, "random_state": 42}
        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)
        r2 = r2_score(y_val, y_pred)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        print(f"RMSE: {rmse}")
        print(f"R2 Score: {r2}")

        # Save the model
        mlflow.sklearn.log_model(rf, "random_forest_model")

if __name__ == '__main__':
    run_train()