# from pmdarima import auto_arima
# import pandas as pd


# def train_sarima(data):
#     model = auto_arima(data, seasonal=True, m=12)
#     return model


# def evaluate_sarima(model, data):
#     predictions = model.predict(n_periods=len(data))
#     mae = sum(abs(predictions - data)) / len(data)
#     return mae


from pmdarima import auto_arima
import pandas as pd

def train_sarima(train_data):
    model = auto_arima(train_data, seasonal=True, m=12)
    return model

def evaluate_sarima(model, test_data):
    predictions = model.predict(n_periods=len(test_data))
    mae = sum(abs(predictions - test_data)) / len(test_data)
    return mae
