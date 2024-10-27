import os
import pandas as pd
import shutil
import csv
from models.var_model import *

# Paths
base_directory = os.path.dirname(__file__)
eval_directory = os.path.join(base_directory, 'evaluations')
data_directory = os.path.join(base_directory, "datasets")

if os.path.exists(eval_directory):
    shutil.rmtree(eval_directory)
    print("folder deleted")

os.makedirs(eval_directory)
csv_files = [
    ("mae.csv", ["fname", "VAR"]),
    ("mape.csv", ["fname", "VAR"]),
    ("mse.csv", ["fname", "VAR"]),
    ("rmse.csv", ["fname", "VAR"]),
]

for filename, header in csv_files:
    filepath = os.path.join(eval_directory, filename)
    with open(filepath, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        print(f"{filename} created")

# Grid search parameters
var_params = {'p_values': range(15), 'd_values': [0, 1, 2]}

csv_mae = os.path.join(eval_directory,  'mae.csv')
csv_mape = os.path.join(eval_directory,  'mape.csv')
csv_mse = os.path.join(eval_directory, 'mse.csv')
csv_rmse = os.path.join(eval_directory,  'rmse.csv')


for filename in os.listdir(data_directory):
    filepath = os.path.join(data_directory, filename)
    print("file -------------", filename)

    df = pd.read_csv(filepath, index_col = 0, parse_dates=True)
    # Step 1: Find the first occurrence of NaN in any column
    first_nan_index = df[df.isna().any(axis=1)].index.min()
    # Step 2: Slice the DataFrame to remove all rows starting from the first NaN
    if pd.notna(first_nan_index):
        df = df.loc[:first_nan_index].iloc[:-1]  # Retain rows before the first NaN row


    try:
        #arima
        results_var = random_search_var(df = df, **var_params)

        new_row_mae = pd.DataFrame(
            [
                [
                    filename,
                    results_var["mae"],

                ]
            ],
            columns=["fname", "VAR"],
        )
        new_row_mape = pd.DataFrame(
            [
              [
                    filename,
                    results_var["mape"],

                ]
            ],
            columns=["fname", "VAR"],
        )
        new_row_mse = pd.DataFrame(
            [
              [
                    filename,
                    results_var["mse"],

                ]
            ],
            columns=["fname", "VAR"],
        )
        new_row_rmse = pd.DataFrame(
            [
              [
                    filename,
                    results_var["rmse"],

                ]
            ],
            columns=["fname", "VAR"],
        )

        new_row_mae.to_csv(
            csv_mae, mode="a", header=False, index=False, lineterminator="\n"
        )
        new_row_mape.to_csv(
            csv_mape, mode="a", header=False, index=False, lineterminator="\n"
        )
        new_row_mse.to_csv(
            csv_mse, mode="a", header=False, index=False, lineterminator="\n"
        )
        new_row_rmse.to_csv(
            csv_rmse, mode="a", header=False, index=False, lineterminator="\n"
        )
    except Exception as e:
        print(f"exception: {e}")
        continue


