# from pmdarima import auto_arima
# import pandas as pd


# def train_arima(data):
#     model = auto_arima(data, seasonal=False)
#     return model


# def evaluate_arima(model, data):
#     predictions = model.predict(n_periods=len(data))
#     mae = sum(abs(predictions - data)) / len(data)
#     return mae


# from pmdarima import auto_arima
# import pandas as pd

# def train_arima(train_data):
#     model = auto_arima(train_data, seasonal=False)
#     return model

# def evaluate_arima(model, test_data):
#     predictions = model.predict(n_periods=len(test_data))
#     mae = sum(abs(predictions - test_data)) / len(test_data)
#     return mae
# ================================================================


# from statsmodels.tsa.arima.model import ARIMA
# from sklearn.metrics import mean_absolute_error
# import itertools

# def grid_search_arima(data, p_values, d_values, q_values):
#     best_score, best_cfg = float("inf"), None 
#     train_size = int(len(data) * 0.8)
#     train, test = data[:train_size], data[train_size:]
    
#     for p, d, q in itertools.product(p_values, d_values, q_values):
#         try:
#             model = ARIMA(train, order=(p, d, q))
#             model_fit = model.fit()
#             predictions = model_fit.forecast(steps=len(test))
#             mae = mean_absolute_error(test, predictions)
            
#             if mae < best_score:
#                 best_score, best_cfg = mae, (p, d, q)
#         except:
#             continue
#     return best_cfg, best_score

# def evaluate_arima(data, best_cfg):
#     train_size = int(len(data) * 0.8)
#     train, test = data[:train_size], data[train_size:]
#     model = ARIMA(train, order=best_cfg)
#     model_fit = model.fit()
#     predictions = model_fit.forecast(steps=len(test))
#     mae = mean_absolute_error(test, predictions)
#     return mae


# ================================================================


from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import itertools
import numpy as np

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def grid_search_arima(data, p_values, d_values, q_values):
    best_score, best_cfg = float("inf"), None 
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]
    
    for p, d, q in itertools.product(p_values, d_values, q_values):
        try:
            model = ARIMA(train, order=(p, d, q))
            model_fit = model.fit()
            predictions = model_fit.forecast(steps=len(test))
            mape = mean_absolute_percentage_error(test, predictions)
            
            if mape < best_score:
                best_score, best_cfg = mape, (p, d, q)
        except:
            continue
    return best_cfg, best_score

def evaluate_arima(data, best_cfg):
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]
    
    model = ARIMA(train, order=best_cfg)
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=len(test))

    # Compute metrics
    mae = mean_absolute_error(test, predictions)
    mape = mean_absolute_percentage_error(test, predictions)
    mse = mean_squared_error(test, predictions)
    rmse = np.sqrt(mse)
    
    print(f"MAE: {mae}")
    print(f"MAPE: {mape}%")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    
    return mae, mape, mse, rmse
