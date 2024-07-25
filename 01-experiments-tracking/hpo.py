import os
import pickle
import click
import mlflow
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

mlflow.set_tracking_uri("http://127.0.0.1:5000")
HPO_EXPERIMENT_NAME = "sgd-hyperopt-bike"
mlflow.set_experiment(HPO_EXPERIMENT_NAME)

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
    "--num_trials",
    default=50,
    help="The number of parameter evaluations for the optimizer to explore"
)
@click.option(
    "--alpha",
    default=None,
    type=float,
    help="Fixed alpha value. If not provided, alpha will be optimized."
)
def run_optimization(data_path: str, num_trials: int, alpha: float):

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    def objective(params):
        if alpha is not None:
            params['alpha'] = alpha

    # Scale the features
    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    def objective(params):
        with mlflow.start_run():
            mlflow.log_params(params)
            
            sgd = SGDRegressor(
                penalty='l2',
                learning_rate='invscaling',
                random_state=42,
                **params
            )

            # Implement batch training
            batch_size = 10000
            for _ in range(params['max_iter']):
                for i in range(0, X_train_scaled.shape[0], batch_size):
                    X_batch = X_train_scaled[i:i+batch_size]
                    y_batch = y_train[i:i+batch_size]
                    sgd.partial_fit(X_batch, y_batch)

            y_pred = sgd.predict(X_val_scaled)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mlflow.log_metric("rmse", rmse)

        return {'loss': rmse, 'status': STATUS_OK}

    search_space = {
        'alpha': hp.loguniform('alpha', np.log(1e-5), np.log(1)),
        'max_iter': scope.int(hp.quniform('max_iter', 10, 100, 1)),
        'eta0': hp.loguniform('eta0', np.log(1e-3), np.log(1)),
    }
  
    rstate = np.random.default_rng(42)  # for reproducible results
    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=Trials(),
        rstate=rstate
    )

if __name__ == '__main__':
    run_optimization()