# from statsmodels.tsa.arima.model import ARIMA
# from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
# import itertools
# import numpy as np

# def grid_search_arima(data, p_values, d_values, q_values):
#     best_score, best_cfg = float("inf"), None 
#     train_size = int(len(data) * 0.8)
#     train, test = data[:train_size], data[train_size:]
    
#     for p, d, q in itertools.product(p_values, d_values, q_values):
#         try:
#             model = ARIMA(train, order=(p, d, q))
#             model_fit = model.fit()
#             predictions = model_fit.forecast(steps=len(test))
#             mape = mean_absolute_percentage_error(test, predictions)
            
#             if mape < best_score:
#                 best_score, best_cfg = mape, (p, d, q)
#         except:
#             continue
#     return best_cfg, best_score

# def evaluate_arima(data, best_cfg):
#     train_size = int(len(data) * 0.8)
#     train, test = data[:train_size], data[train_size:]
    
#     model = ARIMA(train, order=best_cfg)
#     model_fit = model.fit()
#     predictions = model_fit.forecast(steps=len(test))

#     # Compute metrics
#     mae = mean_absolute_error(test, predictions)
#     mape = mean_absolute_percentage_error(test, predictions)
#     mse = mean_squared_error(test, predictions)
#     rmse = np.sqrt(mse)

#     results =  {
#         "mae" : mae, 
#         "mape" : mape, 
#         "mse" : mse, 
#         "rmse": rmse, 
#     }
    
#     return results


from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np

def random_search_arima(data, p_values, d_values, q_values, n_iter=10):
    best_score, best_cfg = float("inf"), None 
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]
    
    for _ in range(n_iter):
        # Randomly select p, d, q values from the provided lists
        p = np.random.choice(p_values)
        d = np.random.choice(d_values)
        q = np.random.choice(q_values)
        print(f"{p}-{d}-{q}")
        
        try:
            model = ARIMA(train, order=(p, d, q), )
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

    results =  {
        "mae" : mae, 
        "mape" : mape, 
        "mse" : mse, 
        "rmse": rmse, 
    }
    
    return results
