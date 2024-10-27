
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np

def random_search_arima(data, p_values, d_values, q_values, n_iter=10):
    data = data.copy(deep=True)
    best_score, best_cfg = float("inf"), None 
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]

    print(f"train rs {train}")
    print(f"test  rs {test}")
    
    for _ in range(n_iter):
        # Randomly select p, d, q values from the provided lists
        p = np.random.choice(p_values)
        d = np.random.choice(d_values)
        q = np.random.choice(q_values)
        

        try:
            print(f"{p}-{d}-{q}")
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
    data = data.copy(deep = True)
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]

    print(f"train eval {train}")
    print(f"test  eval {test}")
    
    model = ARIMA(train, order=best_cfg)
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=len(test))

    # Check for NaNs in predictions
    if np.any(np.isnan(predictions)):
        print(predictions)
        print("Warning: NaNs detected in predictions")
        predictions = np.nan_to_num(predictions, nan=0.0)
        print(predictions)

        # Check for NaNs in test
    if np.any(np.isnan(test)):
        print(test)
        print("Warning: NaNs detected in test")
        test = np.nan_to_num(test, nan=0.0)
        print(test)

    # Compute metrics
    mae = mean_absolute_error(test, predictions)
    mape = mean_absolute_percentage_error(test, predictions)
    mse = mean_squared_error(test, predictions)
    rmse = np.sqrt(mse)

    results =  {
        "mae" : mae, 
        "mape" : mape, 
        "mse" : mse, 
        "rmse": rmse, 
    }
    
    return results
