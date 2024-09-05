from pmdarima import auto_arima
import pandas as pd


def train_arima(data):
    model = auto_arima(data, seasonal=False)
    return model


def evaluate_arima(model, data):
    predictions = model.predict(n_periods=len(data))
    mae = sum(abs(predictions - data)) / len(data)
    return mae
