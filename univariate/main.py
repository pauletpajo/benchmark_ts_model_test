

import os
import pandas as pd
import shutil
import csv
from models.arima_model import *
from models.sarima_model import *
from models.rnn_model import *
from models.lstm_model import *

# Paths
base_directory = os.path.dirname(__file__)
eval_directory = os.path.join(base_directory, 'evaluations')
data_directory = os.path.join(base_directory, "datasets")

if os.path.exists(eval_directory):
    shutil.rmtree(eval_directory)
    print("folder deleted")

os.makedirs(eval_directory)
csv_files = [
    ("mae.csv", ["fname", "ARIMA", "SARIMA", "RNN", "LSTM"]),
    ("mape.csv", ["fname", "ARIMA", "SARIMA", "RNN", "LSTM"]), 
    ("mse.csv", ["fname", "ARIMA", "SARIMA", "RNN", "LSTM"]), 
    ("rmse.csv", ["fname", "ARIMA", "SARIMA", "RNN", "LSTM"]),  
]

for filename, header in csv_files:
    filepath = os.path.join(eval_directory, filename)
    with open(filepath, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        print(f"{filename} created")

# Grid search parameters
arima_params = {'p_values': range(3), 'd_values': [0, 1], 'q_values': range(3)}
sarima_params = {'p_values': range(3), 'd_values': [0, 1], 'q_values': range(3), 
                 'P_values': range(3), 'D_values': [0, 1], 'Q_values': range(3), 'm_values': [12]}
rnn_params = {'units_list': [50, 75, 100], 'epoch_list': [1, 2, 3]}
lstm_params = {'units_list': [32, 64, 128], 'epochs_list': [1, 2, 3]}

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

    

    df_index_removed = pd.read_csv(filepath, usecols=[1])
    # Step 1: Find the first occurrence of NaN in any column
    first_nan_index = df_index_removed[df_index_removed.isna().any(axis=1)].index.min()

    # Step 2: Slice the DataFrame to remove all rows starting from the first NaN
    if pd.notna(first_nan_index):
        df_index_removed = df_index_removed.loc[:first_nan_index].iloc[:-1]  # Retain rows before the first NaN row

    


    try:
        #arima
        best_cfg_arima, _ = random_search_arima(data = df_index_removed, **arima_params)
        if best_cfg_arima is None: 
            best_cfg_arima = (1, 0, 1)
        print(f"best arima config: {best_cfg_arima}")
        results_arima = evaluate_arima(data = df_index_removed, best_cfg=best_cfg_arima)
        print(results_arima)

        #sarima
        best_cfg_sarima, _ = random_search_sarima(data = df, **sarima_params)
        if best_cfg_sarima is None:
            best_cfg_sarima = (1, 1, 1, 1, 1, 1, 12)
        results_sarima = evaluate_sarima(data = df, best_cfg=best_cfg_sarima)

        #rnn
        results_rnn = random_search_rnn(df = df_index_removed, **rnn_params)

        #lstm
        results_lstm = perform_random_search_lstm(df = df_index_removed, **lstm_params)

        new_row_mae = pd.DataFrame(
            [
                [
                    filename,
                    results_arima["mae"],
                    results_sarima["mae"],
                    results_rnn["mae"],
                    results_lstm["mae"],
                ]
            ],
            columns=["fname", "ARIMA", "SARIMA", "RNN", "LSTM"],
        )
        new_row_mape = pd.DataFrame(
            [
                [
                    filename,
                    results_arima["mape"],
                    results_sarima["mape"],
                    results_rnn["mape"],
                    results_lstm["mape"],
                ]
            ],
            columns=["fname", "ARIMA", "SARIMA", "RNN", "LSTM"],
        )
        new_row_mse = pd.DataFrame(
            [
                [
                    filename,
                    results_arima["mse"],
                    results_sarima["mse"],
                    results_rnn["mse"],
                    results_lstm["mse"],
                ]
            ],
            columns=["fname", "ARIMA", "SARIMA", "RNN", "LSTM"],
        )
        new_row_rmse = pd.DataFrame(
            [
                [
                    filename,
                    results_arima["mae"],
                    results_sarima["mae"],
                    results_rnn["mae"],
                    results_lstm["mae"],
                ]
            ],
            columns=["fname", "ARIMA", "SARIMA", "RNN", "LSTM"],
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


