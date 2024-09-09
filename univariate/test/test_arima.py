import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import itertools

def grid_search_arima(data, p_values, d_values, q_values):
    best_score, best_cfg = float("inf"), None
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]
    
    for p, d, q in itertools.product(p_values, d_values, q_values):
        try:
            model = ARIMA(train, order=(p, d, q))
            model_fit = model.fit()
            predictions = model_fit.forecast(steps=len(test))
            mae = mean_absolute_error(test, predictions)
            if mae < best_score:
                best_score, best_cfg = mae, (p, d, q)
        except:
            continue
    return best_cfg, best_score

def evaluate_arima(data, best_cfg):
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]
    model = ARIMA(train, order=best_cfg)
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=len(test))
    mae = mean_absolute_error(test, predictions)
    return mae



data = pd.read_csv('/workspaces/benchmark_ts_model_test/datasets/candy_production.csv', index_col = 0, parse_dates = True)

print(data.head())



