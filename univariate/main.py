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

if os.path.exists(eval_directory):
    shutil.rmtree(eval_directory)
    print("folder deleted")

os.makedirs(eval_directory)
csv_files = [
    ("mae.csv", ["fname", "arima", "sarima", "rnn", "lstm"]),
    ("mape.csv", ["fname", "arima", "sarima", "rnn", "lstm"]), 
    ("mse.csv", ["fname", "arima", "sarima", "rnn", "lstm"]), 
    ("rmse.csv", ["fname", "arima", "sarima", "rnn", "lstm"]),  
]

for filename, header in csv_files:
    filepath = os.path.join(eval_directory, filename)
    with open(filepath, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        print(f"{filename} created")


# Grid search parameters
arima_params = {'p_values': [range(15)], 'd_values': [0, 1], 'q_values': [range(15)]}


sarima_params = {'p_values': [0, 1], 'd_values': [0, 1], 'q_values': [0, 1], 
                 'P_values': [0, 1], 'D_values': [0, 1], 'Q_values': [0, 1], 'm_values': [12]}
rnn_params = {'units': [50, 100], 'epochs': [10, 20]}
lstm_params = {'units': [50, 100], 'epochs': [10, 20]}


data_directory = os.path.join(base_directory, "datasets")
for filename in os.listdir(data_directory):
    filepath = os.path.join(data_directory, filename)

    df = pd.read_csv(filepath, index_col = 0, parse_dates=True)

    csv_mae = os.path.join(eval_directory,  'mae.csv')
    csv_mape = os.path.join(eval_directory,  'mape.csv')
    csv_mse = os.path.join(eval_directory, 'mse.csv')
    csv_rmse = os.path.join(eval_directory,  'rmse.csv')





# ======================================================================================================

# Loop through each dataset
for dataset in os.listdir(DATASETS_PATH):
    data = pd.read_csv(os.path.join(DATASETS_PATH, dataset))

    print(data.head())

    
    # ARIMA
    best_cfg, _ = grid_search_arima(data, **arima_params)
    arima_mae = evaluate_arima(data, best_cfg)
    
    # SARIMA
    best_cfg, _ = grid_search_sarima(data, **sarima_params)
    sarima_mae = evaluate_sarima(data, best_cfg)
    
    # RNN
    best_params, _ = grid_search_rnn(data, rnn_params)
    rnn_mae = evaluate_rnn(data, best_params)
    
    # LSTM
    best_params, _ = grid_search_lstm(data, lstm_params)
    lstm_mae = evaluate_lstm(data, best_params)
    


    
    # Write results to file
    with open(RESULTS_FILE, 'a') as f:
        f.write(f'{dataset},ARIMA,{arima_mae}\n')
        f.write(f'{dataset},SARIMA,{sarima_mae}\n')
        f.write(f'{dataset},RNN,{rnn_mae}\n')
        f.write(f'{dataset},LSTM,{lstm_mae}\n')
