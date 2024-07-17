1. Imports necessary libraries.
Defines utility functions:

dump_pickle: Saves objects as pickle files.

read_dataframe: Reads CSV files, processes timestamps, calculates ride duration, and filters rides between 1 and 180 minutes.


Defines preprocess function:

Extracts hour, day of week, and month from the start time.
Uses DictVectorizer to transform categorical and numerical features.


Main function run_data_prep:

Loads data from three months (January, February, March 2023).
Combines the data and splits it into train, validation, and test sets.

Extracts the target variable (ride duration).
Preprocesses the data using DictVectorizer.
Saves the preprocessed data and DictVectorizer as pickle files.


Uses Click to create a command-line interface.

To use this script:

Ensure you have the required libraries installed:
$ pip install pandas numpy scikit-learn click

Save the script as preprocess_data.py.

Run the script from the command line:
$ python preprocess_data.py --raw_data_path /path/to/raw/data --dest_path /path/to/output


This script assumes that your raw data files are named "YYYYMM-capitalbikeshare-tripdata.csv" and are located in the specified raw_data_path. Adjust the file naming and path handling if your data is structured differently.

This preprocessing script will prepare your bike-sharing data for model training, ensuring consistent feature engineering and vectorization across your train, validation, and test sets.