
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

def apply_differencing(df, d):
    if d == 0:
        return df
    else:
        return df.diff(d).dropna()

def random_search_var(df, d_values=[0, 1, 2], p_values=[1, 2, 3, 4, 5, 10, 15], n_iter=10):
    df_original = df.copy(deep=True)  # Ensure original df isn't modified
    metrics = {}
    train_size = int(len(df) * 0.8)
    best_mape = float('inf')

    for _ in range(n_iter):
        # Randomly select values for d and p
        d = np.random.choice(d_values)
        p = np.random.choice(p_values)
        print(f"{p}-{d}")
        
        # Apply differencing to the data
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

            # Update the best metrics if current MAPE is lower
            if mape < best_mape:
                best_mape = mape
                metrics = {
                    "d": d,
                    "p": p,
                    "mse": mse,
                    "rmse": rmse,
                    "mae": mae,
                    "mape": mape,
                }
        except Exception as e:
            print(f"Model failed for d={d}, p={p}: {e}")

    # Return the best metrics after the search is complete
    return metrics


