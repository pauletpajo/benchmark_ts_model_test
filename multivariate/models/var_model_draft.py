import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import itertools

def apply_differencing(df, d):
    if d == 0:
        return df
    else:
        return df.diff(d).dropna()


def grid_search_var(df, d_values = [0, 1, 2],  p = 15):
    df = df.copy(deep=True)

    metrics = {}

    train_size = int(len(df) * 0.8)



    counter = True
    for d in d_values: 
        #apply differening on the df, so that we don't have to offset the forecast to compute the metrics. 
        df = apply_differencing(df, d)
        df_train_diff, df_test_diff = df[:train_size], df[train_size:]

        try:
            # Fit VAR model on training data
            model = VAR(df_train_diff)
            fitted_model = model.fit(p)

            # Forecast future values for the test set
            forecast = fitted_model.forecast(df_train_diff.values[-fitted_model.k_ar:], steps=len(df_test_diff))
            
            forecast_df = pd.DataFrame(forecast, index=df_test_diff.index, columns=df.columns)

            mse = mean_squared_error(df_test_diff[:,-1], forecast_df[:, -1])
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(df_test_diff[:,-1], forecast_df[:, -1])
            mape = mean_absolute_percentage_error(df_test_diff[:,-1], forecast_df[:, -1])


            if counter == True: 
                metrics = {
                    "mse": mse, 
                    "rmse": rmse, 
                    "mae": mae, 
                    "mape": mape, 
                }
                counter = False
            else:
                if mape < metrics['mape']:
                    metrics = {
                        "mse": mse, 
                        "rmse": rmse, 
                        "mae": mae, 
                        "mape": mape, 
                    }
        except Exception as e:
            print(f"Model failed: {e}")

    # Return the best parameters and MSE after the search is complete
    return metrics



# Load a dataset for testing (replace 'your_dataset.csv' with the actual dataset file)
df = pd.read_csv('datasets/apple2.csv', index_col=0, parse_dates=True)

# Call the grid search function and get the best parameters
metrics = grid_search_var(df)
print(metrics)


# import pandas as pd
# import numpy as np
# from statsmodels.tsa.api import VAR
# from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
# import itertools

# def apply_differencing(df, d):
#     if d == 0:
#         return df
#     else:
#         return df.diff(d).dropna()

# def grid_search_var(df, d_values=[0, 1, 2], p=15):
#     df = df.copy(deep=True)
    
#     # Initialize an empty dictionary to hold the best metrics
#     best_metrics = None

#     train_size = int(len(df) * 0.8)

#     for d in d_values: 
#         # Apply differencing
#         df_diff = apply_differencing(df, d)

#         # Split train and test sets from the differenced dataframe
#         train, test = df_diff.iloc[:train_size, :], df_diff.iloc[train_size:, :]

#         try:
#             # Fit VAR model on training data
#             model = VAR(train)
#             fitted_model = model.fit(p)

#             # Forecast future values for the test set
#             forecast = fitted_model.forecast(train.values[-fitted_model.k_ar:], steps=len(test))
#             forecast_df = pd.DataFrame(forecast, index=test.index, columns=df.columns)


#             # Calculate evaluation metrics (MSE, RMSE, MAE, MAPE) for the last column (target)
#             mse = mean_squared_error(test.iloc[:, -1], forecast_df.iloc[:, -1])
#             rmse = np.sqrt(mse)
#             mae = mean_absolute_error(test.iloc[:, -1], forecast_df.iloc[:, -1])
#             mape = mean_absolute_percentage_error(test.iloc[:, -1], forecast_df.iloc[:, -1])




#             # Update the best metrics if this iteration has a lower MAPE
#             if best_metrics is None or mape < best_metrics['mape']:
#                 best_metrics = {
#                     "d": d,
#                     "p": p,
#                     "mse": mse,
#                     "rmse": rmse,
#                     "mae": mae,
#                     "mape": mape
#                 }
#         except Exception as e:
#             print(f"Model failed for d={d}: {e}")

#     # Return the best metrics after the grid search is complete
#     return best_metrics

# # Load a dataset for testing (replace 'your_dataset.csv' with the actual dataset file)
# df = pd.read_csv('datasets/apple2.csv', index_col=0, parse_dates=True)

# # Call the grid search function and get the best metrics
# best_metrics = grid_search_var(df)

# # Print the best parameters and evaluation metrics
# if best_metrics:
#     print(f"Best Parameters: d={best_metrics['d']}, p={best_metrics['p']}")
#     print(f"Metrics: MSE={best_metrics['mse']}, RMSE={best_metrics['rmse']}, MAE={best_metrics['mae']}, MAPE={best_metrics['mape']}")
# else:
#     print("No model succeeded during grid search.")
