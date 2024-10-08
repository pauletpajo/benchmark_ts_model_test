import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


def apply_differencing(df, d):
    if d == 0:
        return df
    else:
        return df.diff(d).dropna()


def grid_search_var(df, d_values=[0, 1, 2], p=15):
    df_original = df.copy(deep=True)  # Ensure original df isn't modified
    metrics = {}

    train_size = int(len(df) * 0.8)
    counter = True

    for d in d_values:
        # Apply differencing on the df so that we don't have to offset the forecast to compute the metrics.
        df_diff = apply_differencing(df_original, d)
        df_train_diff, df_test_diff = df_diff[:train_size], df_diff[train_size:]

        try:
            # Fit VAR model on training data
            model = VAR(df_train_diff)
            fitted_model = model.fit(p)

            # Forecast future values for the test set
            forecast = fitted_model.forecast(df_train_diff.values[-fitted_model.k_ar:], steps=len(df_test_diff))
            forecast_df = pd.DataFrame(forecast, index=df_test_diff.index, columns=df.columns)

            # Calculate error metrics on the target column (last column)
            mse = mean_squared_error(df_test_diff.iloc[:, -1], forecast_df.iloc[:, -1])
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(df_test_diff.iloc[:, -1], forecast_df.iloc[:, -1])
            mape = mean_absolute_percentage_error(df_test_diff.iloc[:, -1], forecast_df.iloc[:, -1])

            # Store the best result (minimum MAPE)
            if counter:
                metrics = {
                    "mse": mse,
                    "rmse": rmse,
                    "mae": mae,
                    "mape": mape,
                }
                counter = False  # Set counter to False after the first iteration
            else:
                if mape < metrics['mape']:
                    metrics = {
                        "mse": mse,
                        "rmse": rmse,
                        "mae": mae,
                        "mape": mape,
                    }
        except Exception as e:
            print(f"Model failed for d={d}, p={p}: {e}")

    # Return the best metrics after the search is complete
    return metrics


# Load a dataset for testing (replace 'your_dataset.csv' with the actual dataset file)
df = pd.read_csv('datasets/apple2.csv', index_col=0, parse_dates=True)

# Call the grid search function and get the best parameters
metrics = grid_search_var(df)
print(metrics)

