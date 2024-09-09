import os
import pandas as pd

from models.arima_model import *
from models.sarima_model import *
from models.rnn_model import *
from models.lstm_model import *



# Paths
DATASETS_PATH = 'datasets/'
RESULTS_FILE = 'results/evaluations.csv'

# Grid search parameters
arima_params = {'p_values': [0, 1, 2], 'd_values': [0, 1], 'q_values': [0, 1, 2]}
sarima_params = {'p_values': [0, 1], 'd_values': [0, 1], 'q_values': [0, 1], 
                 'P_values': [0, 1], 'D_values': [0, 1], 'Q_values': [0, 1], 'm_values': [12]}
rnn_params = {'units': [50, 100], 'epochs': [10, 20]}
lstm_params = {'units': [50, 100], 'epochs': [10, 20]}

# Initialize results file
if not os.path.exists('results'):
    os.makedirs('results')
if not os.path.isfile(RESULTS_FILE):
    with open(RESULTS_FILE, 'w') as f:
        f.write('Dataset,Model,MAE\n')

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
