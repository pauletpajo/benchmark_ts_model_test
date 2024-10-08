
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# import itertools
# import numpy as np
# import pandas as pd


# def mean_absolute_percentage_error(y_true, y_pred):
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



# def grid_search_sarima(data, p_values, d_values, q_values, P_values, D_values, Q_values, m_values):
#     best_score, best_cfg = float("inf"), None
#     train_size = int(len(data) * 0.8)
#     train, test = data[:train_size], data[train_size:]
    
#     for p, d, q, P, D, Q, m in itertools.product(p_values, d_values, q_values, P_values, D_values, Q_values, m_values):
#         try:
#             model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, m))
#             model_fit = model.fit(disp=False)
#             predictions = model_fit.forecast(steps=len(test))
#             mape = mean_absolute_percentage_error(test, predictions)


#             if mape < best_score:
#                 best_score, best_cfg = mape, (p, d, q, P, D, Q, m)
#         except:
#             continue
#     return best_cfg, best_score

# def evaluate_sarima(data, best_cfg):
#     train_size = int(len(data) * 0.8)
#     train, test = data[:train_size], data[train_size:]


#     p, d, q, P, D, Q, m = best_cfg
#     model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, m))
#     model_fit = model.fit(disp=False)
#     predictions = model_fit.forecast(steps=len(test))

#    # Compute metrics
#     mae = mean_absolute_error(test, predictions)
#     mape = mean_absolute_percentage_error(test, predictions)
#     mse = mean_squared_error(test, predictions)
#     rmse = np.sqrt(mse)
    
#     print(f"MAE: {mae}")
#     print(f"MAPE: {mape}%")
#     print(f"MSE: {mse}")
#     print(f"RMSE: {rmse}")
    
#     return mae, mape, mse, rmse


# # data = pd.read_csv('/content/drive/MyDrive/data_ts/candy_production.csv', index_col=0, parse_dates=True)
# data = pd.read_csv('datasets/candy_production.csv', index_col=0, parse_dates=True)

# sarima_params = {'p_values': [0, 1], 'd_values': [0, 1], 'q_values': [0, 1], 
#                  'P_values': [0, 1], 'D_values': [0, 1], 'Q_values': [0, 1], 'm_values': [12]}


# # SARIMA
# best_cfg, _ = grid_search_sarima(data, **sarima_params)
# sarima_mae = evaluate_sarima(data, best_cfg)
    

# import pandas as pd
# import numpy as np
# from statsmodels.tsa.api import VAR
# from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
# import itertools

# def apply_differencing(df, d):
#     if d == 0:
#         return df
#     else:
#         return df.diff(d).dropna()

# def grid_search_var(df, d_values=[0, 1, 2], p=15):
#     df = df.copy(deep=True)
    
#     # Initialize an empty dictionary to hold the best metrics
#     best_metrics = None

#     train_size = int(len(df) * 0.8)

#     for d in d_values: 
#         # Apply differencing
#         df_diff = apply_differencing(df, d)

#         # Split train and test sets from the differenced dataframe
#         train, test = df_diff.iloc[:train_size, :], df_diff.iloc[train_size:, :]

#         try:
#             # Fit VAR model on training data
#             model = VAR(train)
#             fitted_model = model.fit(p)

#             # Forecast future values for the test set
#             forecast = fitted_model.forecast(train.values[-fitted_model.k_ar:], steps=len(test))
#             forecast_df = pd.DataFrame(forecast, index=test.index, columns=df.columns)


#             # Calculate evaluation metrics (MSE, RMSE, MAE, MAPE) for the last column (target)
#             mse = mean_squared_error(test.iloc[:, -1], forecast_df.iloc[:, -1])
#             rmse = np.sqrt(mse)
#             mae = mean_absolute_error(test.iloc[:, -1], forecast_df.iloc[:, -1])
#             mape = mean_absolute_percentage_error(test.iloc[:, -1], forecast_df.iloc[:, -1])

#             # Update the best metrics if this iteration has a lower MAPE
#             if best_metrics is None or mape < best_metrics['mape']:
#                 best_metrics = {
#                     "d": d,
#                     "p": p,
#                     "mse": mse,
#                     "rmse": rmse,
#                     "mae": mae,
#                     "mape": mape
#                 }
#         except Exception as e:
#             print(f"Model failed for d={d}: {e}")

#     # Return the best metrics after the grid search is complete
#     return best_metrics

# # Load a dataset for testing (replace 'your_dataset.csv' with the actual dataset file)
# df = pd.read_csv('datasets/apple2.csv', index_col=0, parse_dates=True)

# # Call the grid search function and get the best metrics
# best_metrics = grid_search_var(df)

# # Print the best parameters and evaluation metrics
# if best_metrics:
#     print(f"Best Parameters: d={best_metrics['d']}, p={best_metrics['p']}")
#     print(f"Metrics: MSE={best_metrics['mse']}, RMSE={best_metrics['rmse']}, MAE={best_metrics['mae']}, MAPE={best_metrics['mape']}")
# else:
#     print("No model succeeded during grid search.")



# import pandas as pd
# import numpy as np
# from statsmodels.tsa.api import VAR
# from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


# def apply_differencing(df, d):
#     if d == 0:
#         return df
#     else:
#         return df.diff(d).dropna()


# def grid_search_var(df, d_values=[0, 1, 2], p=15):
#     df_original = df.copy(deep=True)  # Ensure original df isn't modified
#     metrics = {}

#     train_size = int(len(df) * 0.8)
#     counter = True

#     for d in d_values:
#         # Apply differencing on the df so that we don't have to offset the forecast to compute the metrics.
#         df_diff = apply_differencing(df_original, d)
#         df_train_diff, df_test_diff = df_diff[:train_size], df_diff[train_size:]

#         try:
#             # Fit VAR model on training data
#             model = VAR(df_train_diff)
#             fitted_model = model.fit(p)

#             # Forecast future values for the test set
#             forecast = fitted_model.forecast(df_train_diff.values[-fitted_model.k_ar:], steps=len(df_test_diff))
#             forecast_df = pd.DataFrame(forecast, index=df_test_diff.index, columns=df.columns)

#             # Calculate error metrics on the target column (last column)
#             mse = mean_squared_error(df_test_diff.iloc[:, -1], forecast_df.iloc[:, -1])
#             rmse = np.sqrt(mse)
#             mae = mean_absolute_error(df_test_diff.iloc[:, -1], forecast_df.iloc[:, -1])
#             mape = mean_absolute_percentage_error(df_test_diff.iloc[:, -1], forecast_df.iloc[:, -1])

#             # Store the best result (minimum MAPE)
#             if counter:
#                 metrics = {
#                     "mse": mse,
#                     "rmse": rmse,
#                     "mae": mae,
#                     "mape": mape,
#                 }
#                 counter = False  # Set counter to False after the first iteration
#             else:
#                 if mape < metrics['mape']:
#                     metrics = {
#                         "mse": mse,
#                         "rmse": rmse,
#                         "mae": mae,
#                         "mape": mape,
#                     }
#         except Exception as e:
#             print(f"Model failed for d={d}, p={p}: {e}")

#     # Return the best metrics after the search is complete
#     return metrics


# # Load a dataset for testing (replace 'your_dataset.csv' with the actual dataset file)
# df = pd.read_csv('datasets/apple2.csv', index_col=0, parse_dates=True)

# # Call the grid search function and get the best parameters
# metrics = grid_search_var(df)
# print(metrics)




import pandas as pd
import numpy as np
import math
import itertools
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Function to split data into X (inputs) and Y (targets)
def get_XY(data, time_steps):
    Y_ind = np.arange(time_steps, len(data), time_steps)
    Y = data[Y_ind]
    rows_x = len(Y)
    X = data[range(time_steps * rows_x)]
    X = np.reshape(X, (rows_x, time_steps, 1))    
    return X, Y



# Function to build and compile the RNN model
def build_rnn(time_steps=12, units=3, dense_units=1, activation=['tanh', 'tanh']):
    model = Sequential()
    model.add(SimpleRNN(units, input_shape=(time_steps, 1), activation=activation[0], return_sequences=False))
    model.add(Dense(dense_units, activation=activation[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Function to train the RNN model
def train_rnn(train_data, time_steps=12, units=3, epochs=10, batch_size=1):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(np.array(train_data).astype('float32')).flatten()
    
    trainX, trainY = get_XY(data, time_steps)
    
    model = build_rnn(time_steps, units)
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=0)
    
    return model, scaler

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# Function to evaluate the trained RNN model
def evaluate_rnn(model, test_data, scaler, time_steps=12):
    test_data = scaler.transform(np.array(test_data).astype('float32')).flatten()
    testX, testY = get_XY(test_data, time_steps)
    
    test_predict = model.predict(testX)
    
    test_predict = test_predict.reshape(-1)
    testY = testY.reshape(-1)
    
    # test_rmse = math.sqrt(mean_squared_error(testY, test_predict))
    test_mape = mean_absolute_percentage_error(testY, test_predict)
    return test_mape

def evaluate_final_model(model, test_data, scaler, time_steps=12):
    test_data = scaler.transform(np.array(test_data).astype('float32')).flatten()
    testX, testY = get_XY(test_data, time_steps)
    
    test_predict = model.predict(testX)
    
    test_predict = test_predict.reshape(-1)
    testY = testY.reshape(-1)
    
    mae = mean_absolute_error(testY, test_predict)
    mse = mean_squared_error(testY, test_predict)
    rmse = math.sqrt(mean_squared_error(testY, test_predict))
    mape = mean_absolute_percentage_error(testY, test_predict)
    return mae, mape, mse, rmse


# Function to perform grid search for the number of units
def grid_search(df, units_list, epoch_list, time_steps=12):

    split_percent = 0.8
    split = int(len(df) * split_percent)
    train_data = df[:split]
    test_data = df[split:]

    best_mape = float('inf')
    best_units = None
    best_model = None
    best_scaler = None

    for units, epochs in itertools.product(units_list, epoch_list):
        print(f"Training with units={units}...")
        model, scaler = train_rnn(train_data=train_data,time_steps= time_steps,units= units, epochs=epochs)
        test_mape = evaluate_rnn(model, test_data, scaler, time_steps)
        print(f"Test mape with units={units}: {test_mape:.3f}")
        
        if test_mape < best_mape:
            best_rmse = test_mape
            best_units = units
            best_model = model
            best_scaler = scaler

    print(f"Best mape: {best_rmse:.3f}")
    print(f"Best units: {best_units}")


    mae, mape, mse, rmse = evaluate_final_model(best_model, test_data, best_scaler)

    return mae, mape, mse, rmse




def main():
    # Load data
    # sunspots_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv'
    # df = pd.read_csv(sunspots_url, usecols=[1], engine='python')
    df = pd.read_csv('datasets/candy_production.csv', index_col=0, parse_dates=True)

    
    # Define list of units to search
    units_list = [3, 4, 5]
    epoch_list = [3, 4, 5, 10]

    # Perform grid search to find the best number of units
    mae, mape, mse, rmse = grid_search(df, units_list, epoch_list=epoch_list)

    print(f"mae: {mae}")
    print(f"mape: {mape}")
    print(f"mse: {mse}")
    print(f"rmse: {rmse}")
if __name__ == "__main__":
    main()

