Changed the MLflow experiment name: "bike-sharing-random-forest".

num_trees: Number of trees in the Random Forest (default 100)
max_depth: Maximum depth of the trees (default 10)


In the run_train function:

Created a params dictionary with the Random Forest parameters.
Used these params when initializing the RandomForestRegressor.


Added R2 score calculation alongside RMSE for a more comprehensive model evaluation.
Logged both RMSE and R2 score as metrics in MLflow.
Added print statements to display RMSE and R2 score in the console.
Used mlflow.sklearn.log_model to save the trained model in MLflow.

To use this script:

Ensure you have MLflow installed:
$ pip install mlflow

Save the script as train.py.

Run the script from the command line:
$ python train.py --data_path /path/to/processed/data --num_trees 150 --max_depth 15


This script will:

* Load the preprocessed training and validation data
* Train a Random Forest model with the specified parameters
* Evaluate the model using RMSE and R2 score
* Log the parameters, metrics, and model in MLflow

You can then use the MLflow UI to compare different runs with various parameters:
$ mlflow ui

This will start the MLflow UI, which you can access in your web browser to view and compare your experiments.