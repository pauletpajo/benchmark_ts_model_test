# from pmdarima import auto_arima
# import pandas as pd


# def train_sarima(data):
#     model = auto_arima(data, seasonal=True, m=12)
#     return model


# def evaluate_sarima(model, data):
#     predictions = model.predict(n_periods=len(data))
#     mae = sum(abs(predictions - data)) / len(data)
#     return mae


# from pmdarima import auto_arima
# import pandas as pd

# def train_sarima(train_data):
#     model = auto_arima(train_data, seasonal=True, m=12)
#     return model

# def evaluate_sarima(model, test_data):
#     predictions = model.predict(n_periods=len(test_data))
#     mae = sum(abs(predictions - test_data)) / len(test_data)
#     return mae



from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
import itertools

def grid_search_sarima(data, p_values, d_values, q_values, P_values, D_values, Q_values, m_values):
    best_score, best_cfg = float("inf"), None
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]
    
    for p, d, q, P, D, Q, m in itertools.product(p_values, d_values, q_values, P_values, D_values, Q_values, m_values):
        try:
            model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, m))
            model_fit = model.fit(disp=False)
            predictions = model_fit.forecast(steps=len(test))
            mae = mean_absolute_error(test, predictions)
            if mae < best_score:
                best_score, best_cfg = mae, (p, d, q, P, D, Q, m)
        except:
            continue
    return best_cfg, best_score

def evaluate_sarima(data, best_cfg):
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]
    p, d, q, P, D, Q, m = best_cfg
    model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, m))
    model_fit = model.fit(disp=False)
    predictions = model_fit.forecast(steps=len(test))
    mae = mean_absolute_error(test, predictions)
    return mae
