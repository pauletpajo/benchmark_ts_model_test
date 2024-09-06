# from pmdarima import auto_arima
# import pandas as pd


# def train_arima(data):
#     model = auto_arima(data, seasonal=False)
#     return model


# def evaluate_arima(model, data):
#     predictions = model.predict(n_periods=len(data))
#     mae = sum(abs(predictions - data)) / len(data)
#     return mae


from pmdarima import auto_arima
import pandas as pd

def train_arima(train_data):
    model = auto_arima(train_data, seasonal=False)
    return model

def evaluate_arima(model, test_data):
    predictions = model.predict(n_periods=len(test_data))
    mae = sum(abs(predictions - test_data)) / len(test_data)
    return mae
