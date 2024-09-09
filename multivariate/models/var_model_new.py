import pandas as pd
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error
import itertools

def apply_differencing(df, d):
    if d == 0:
        return df
    else:
        return df.diff(d).dropna()


def grid_search_var(df, d_values = [0, 1, 2],  p_values = range(1, 11)):

    # Initialize best parameters
    best_mse = float('inf')
    best_params = {'d': None, 'p': None}

    train_size = int(len(df) * 0.8)
    df_train, df_test = df[:train_size], df[train_size:]

    # Initialize best parameters
    best_mse = float('inf')
    best_params = {'d': None, 'p': None}

    for d, p in itertools.product(d_values, p_values): 
    # Apply differencing to the training data
        df_train_diff = apply_differencing(df_train, d)
        
        # Adjust the length of the test set based on differencing
        df_test_diff = apply_differencing(df_test, d)
    
        try:
            # Fit VAR model on training data
            model = VAR(df_train_diff)
            results = model.fit(p)
            
            # Forecast for the length of the test set
            lag_order = p
            forecast_input = df_train_diff.values[-lag_order:]
            
            # Forecast future values for the test set
            forecast = results.forecast(y=forecast_input, steps=len(df_test_diff))
            
            # Offset forecast by 'd' to align with the original test set
            df_test_aligned = df_test[d:].values  # Skip first 'd' rows in the original test set
            
            # Calculate MSE between the aligned test set and the forecast
            mse = mean_squared_error(df_test_aligned, forecast)
            

            # Check if this is the best model so far
            if mse < best_mse:
                best_mse = mse
                best_params = {'d': d, 'p': p}
                
        except Exception as e:
            print(f"Model failed for d={d}, p={p}: {e}")

    # Return the best parameters and MSE after the search is complete
    return best_params, best_mse



# Load a dataset for testing (replace 'your_dataset.csv' with the actual dataset file)
df = pd.read_csv('datasets/apple2.csv', index_col=0, parse_dates=True)

# Call the grid search function and get the best parameters
best_params, best_mse = grid_search_var(df)

# Print the best parameters and MSE
print(f"Best Parameters: d={best_params['d']}, p={best_params['p']}, MSE={best_mse}")