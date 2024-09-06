# import os
# import pandas as pd
# # from models.arima_model import train_arima, evaluate_arima
# # from models.sarima_model import train_sarima, evaluate_sarima
# # from models.rnn_model import train_rnn, evaluate_rnn
# # from models.lstm_model import train_lstm, evaluate_lstm

# # Paths
# DATASETS_PATH = "datasets/"
# RESULTS_FILE = "results/evaluation_metrics.csv"

# # Initialize results file
# if not os.path.exists("results"):
#     os.makedirs("results")
# if not os.path.isfile(RESULTS_FILE):
#     with open(RESULTS_FILE, "w") as f:
#         f.write("Dataset,Model,MAE\n")

# # Loop through each dataset
# for dataset in os.listdir(DATASETS_PATH):
#     data = pd.read_csv(os.path.join(DATASETS_PATH, dataset))

#     print(data.head())

#     # # Train and evaluate ARIMA
#     # arima_model = train_arima(data)
#     # arima_mae = evaluate_arima(arima_model, data)

#     # # Train and evaluate SARIMA
#     # sarima_model = train_sarima(data)
#     # sarima_mae = evaluate_sarima(sarima_model, data)

#     # # Train and evaluate RNN
#     # rnn_model = train_rnn(data)
#     # rnn_mae = evaluate_rnn(rnn_model, data)

#     # # Train and evaluate LSTM
#     # lstm_model = train_lstm(data)
#     # lstm_mae = evaluate_lstm(lstm_model, data)

#     # # Write results to file
#     # with open(RESULTS_FILE, "a") as f:
#     #     f.write(f"{dataset},ARIMA,{arima_mae}\n")
#     #     f.write(f"{dataset},SARIMA,{sarima_mae}\n")
#     #     f.write(f"{dataset},RNN,{rnn_mae}\n")
#     #     f.write(f"{dataset},LSTM,{lstm_mae}\n")

#     arima_mae = 0.2342
#     dataset = "dataset1"

#     with open(RESULTS_FILE, "a") as f:
#         f.write(f"{dataset},ARIMA,{arima_mae}\n")

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from models.arima_model import train_arima, evaluate_arima
from models.sarima_model import train_sarima, evaluate_sarima
from models.rnn_model import train_rnn, evaluate_rnn
from models.lstm_model import train_lstm, evaluate_lstm

# Paths
DATASETS_PATH = 'datasets/'
RESULTS_FILE = 'results/evaluation_metrics.csv'

# Initialize results file
if not os.path.exists('results'):
    os.makedirs('results')
if not os.path.isfile(RESULTS_FILE):
    with open(RESULTS_FILE, 'w') as f:
        f.write('Dataset,Model,MAE\n')

# Loop through each dataset
for dataset in os.listdir(DATASETS_PATH):
    data = pd.read_csv(os.path.join(DATASETS_PATH, dataset))
    
    # Splitting data into training and testing sets (80% training, 20% testing)
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)
    
    # Train and evaluate ARIMA
    arima_model = train_arima(train_data)
    arima_mae = evaluate_arima(arima_model, test_data)
    
    # Train and evaluate SARIMA
    sarima_model = train_sarima(train_data)
    sarima_mae = evaluate_sarima(sarima_model, test_data)
    
    # Train and evaluate RNN
    rnn_model = train_rnn(train_data)
    rnn_mae = evaluate_rnn(rnn_model, test_data)
    
    # Train and evaluate LSTM
    lstm_model = train_lstm(train_data)
    lstm_mae = evaluate_lstm(lstm_model, test_data)
    
    # Write results to file
    with open(RESULTS_FILE, 'a') as f:
        f.write(f'{dataset},ARIMA,{arima_mae}\n')
        f.write(f'{dataset},SARIMA,{sarima_mae}\n')
        f.write(f'{dataset},RNN,{rnn_mae}\n')
        f.write(f'{dataset},LSTM,{lstm_mae}\n')


