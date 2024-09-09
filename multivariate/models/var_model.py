# import pandas as pd
# from statsmodels.tsa.api import VAR
# from sklearn.metrics import mean_squared_error

# # Example DataFrame
# df = pd.read_csv('datasets/apple2.csv', index_col = 0, parse_dates = True)  # Your multivariate time series data

# # Function to apply differencing
# def apply_differencing(df, d):
#     if d == 0:
#         return df  # No differencing if d is 0
#     else:
#         return df.diff(d).dropna()

# # Split data into train and test sets
# # You can adjust the split ratio as needed
# train_size = int(len(df) * 0.8)  # 80% for training
# df_train, df_test = df[:train_size], df[train_size:]

# # Search space for differencing order (d) and lag (p)
# d_values = [0, 1, 2]  # Differencing orders to try
# p_values = range(1, 11)  # Lags to try



# best_mse = float('inf')
# best_params = {'d': None, 'p': None}

# for d in d_values:
#     # Apply differencing to the training data
#     df_train_diff = apply_differencing(df_train, d)
    
#     for p in p_values:
#         try:
#             # Fit VAR model on training data
#             model = VAR(df_train_diff)
#             results = model.fit(p)
            
#             # Forecast for test set length
#             lag_order = results.k_ar
#             forecast_input = df_train_diff.values[-lag_order:]
            
#             # Forecast future values for the test set
#             forecast = results.forecast(y=forecast_input, steps=len(df_test))
            
#             # Apply differencing to test data to compare
#             df_test_diff = apply_differencing(df_test, d)
            
#             # Calculate MSE on the differenced test data
#             mse = mean_squared_error(df_test_diff.iloc[:len(forecast)], forecast)
            
#             # Check if this is the best model so far
#             if mse < best_mse:
#                 best_mse = mse
#                 best_params = {'d': d, 'p': p}
#         except Exception as e:
#             print(f"Model failed for d={d}, p={p}: {e}")

# print(f"Best Parameters: d={best_params['d']}, p={best_params['p']}, MSE={best_mse}")





# import pandas as pd
# from statsmodels.tsa.api import VAR
# from sklearn.metrics import mean_squared_error

# # Load DataFrame
# df = pd.read_csv('datasets/apple2.csv', index_col=0, parse_dates=True)

# # Function to apply differencing
# def apply_differencing(df, d):
#     if d == 0:
#         return df
#     else:
#         return df.diff(d).dropna()

# # Split data into train and test sets
# train_size = int(len(df) * 0.8)
# df_train, df_test = df[:train_size], df[train_size:]

# # Initialize best parameters
# best_mse = float('inf')
# best_params = {'d': None, 'p': None}

# # Search space for differencing order (d) and lag (p)
# d_values = [0, 1, 2]  # Differencing orders to try
# p_values = range(1, 11)  # Lags to try

# for d in d_values:
#     # Apply differencing to the training data
#     df_train_diff = apply_differencing(df_train, d)
    
#     # Adjust the length of the test set based on differencing
#     df_test_diff = apply_differencing(df_test, d)
    
#     for p in p_values:
#         try:
#             # Fit VAR model on training data
#             model = VAR(df_train_diff)
#             results = model.fit(p)
            
#             # Forecast for the length of the test set
#             lag_order = results.k_ar
#             forecast_input = df_train_diff.values[-lag_order:]
            
#             # Forecast future values for the test set
#             forecast = results.forecast(y=forecast_input, steps=len(df_test_diff))
            
#             # Calculate MSE on the differenced test data
#             mse = mean_squared_error(df_test_diff, forecast)
            
#             # Check if this is the best model so far
#             if mse < best_mse:
#                 best_mse = mse
#                 best_params = {'d': d, 'p': p}
                
#         except Exception as e:
#             print(f"Model failed for d={d}, p={p}: {e}")

# print(f"Best Parameters: d={best_params['d']}, p={best_params['p']}, MSE={best_mse}")



import pandas as pd
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error

# Load DataFrame
df = pd.read_csv('datasets/apple2.csv', index_col=0, parse_dates=True)

# Function to apply differencing
def apply_differencing(df, d):
    if d == 0:
        return df
    else:
        return df.diff(d).dropna()

# Split data into train and test sets
train_size = int(len(df) * 0.8)
df_train, df_test = df[:train_size], df[train_size:]

# Initialize best parameters
best_mse = float('inf')
best_params = {'d': None, 'p': None}

# Search space for differencing order (d) and lag (p)
d_values = [0, 1, 2]  # Differencing orders to try
p_values = range(1, 11)  # Lags to try

for d in d_values:
    # Apply differencing to the training data
    df_train_diff = apply_differencing(df_train, d)
    
    # Adjust the length of the test set based on differencing
    df_test_diff = apply_differencing(df_test, d)
    
    for p in p_values:
        try:
            # Fit VAR model on training data
            model = VAR(df_train_diff)
            results = model.fit(p)
            
            # Forecast for the length of the test set
            lag_order = results.k_ar
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

print(f"Best Parameters: d={best_params['d']}, p={best_params['p']}, MSE={best_mse}")
