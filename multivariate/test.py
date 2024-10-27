
import pandas as pd
import numpy as np
from models.var_model import *

df = pd.read_csv("/workspaces/benchmark_ts_model_test/multivariate/datasets/D2054_D2055_D2056.csv", index_col=0, parse_dates = True)
# Step 1: Find the first occurrence of NaN in any column
first_nan_index = df[df.isna().any(axis=1)].index.min()

# Step 2: Slice the DataFrame to remove all rows starting from the first NaN
if pd.notna(first_nan_index):
    df = df.loc[:first_nan_index].iloc[:-1]  # Retain rows before the first NaN row


var_params = {'p_values': range(15), 'd_values': [0, 1, 2]}

results = random_search_var(df, **var_params
)
print(results)
