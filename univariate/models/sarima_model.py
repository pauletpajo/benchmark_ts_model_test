

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np

def random_search_sarima(data, p_values, d_values, q_values, P_values, D_values, Q_values, m_values, n_iter=10):
    data = data.copy(deep = True)
    best_score, best_cfg = float("inf"), None
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]
    
    for _ in range(n_iter):
        
        # Randomly select each parameter from the provided lists
        p = np.random.choice(p_values)
        d = np.random.choice(d_values)
        q = np.random.choice(q_values)
        P = np.random.choice(P_values)
        D = np.random.choice(D_values)
        Q = np.random.choice(Q_values)
        m = np.random.choice(m_values)
        print(f"{p}-{d}-{q}-{P}-{D}-{Q}-{m}")
        
        try:
            model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, m))
            model_fit = model.fit(disp=False)
            predictions = model_fit.forecast(steps=len(test))
            mape = mean_absolute_percentage_error(test, predictions)

            if mape < best_score:
                best_score, best_cfg = mape, (p, d, q, P, D, Q, m)
        except:
            continue
    
    return best_cfg, best_score

def evaluate_sarima(data, best_cfg):
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]
    
    # Unpack best configuration
    p, d, q, P, D, Q, m = best_cfg
    model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, m))
    model_fit = model.fit(disp=False)
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
