# import pandas as pd
# import numpy as np
# from models.rnn_model import *

# df = pd.read_csv("/workspaces/benchmark_ts_model_test/univariate/datasets/D1.csv", usecols=[1])
# # Define list of units to search
# units_list = [3, 4, 5]
# epoch_list = [1, 2, 3, 4]

# results = random_search_rnn(df = df, units_list=units_list, epoch_list=epoch_list)
# print(results)

# ================================================================================================

# import pandas as pd
# import numpy as np
# from models.lstm_model import *

# df = pd.read_csv("/workspaces/benchmark_ts_model_test/univariate/datasets/D1.csv", usecols=[1])
# # Define list of units to search
# units_list = [3, 4, 5]
# epoch_list = [2, 3]

# results = perform_random_search_lstm(df = df, units_list=units_list, epochs_list=epoch_list)
# print(results)

import pandas as pd
import numpy as np
from models.arima_model import *

df = pd.read_csv("/workspaces/benchmark_ts_model_test/univariate/datasets/D1.csv", usecols=[1])
# Define list of units to search
arima_params = {'p_values': range(7), 'd_values': [0, 1], 'q_values': range(7)}

best_cfg, _ = random_search_arima(data = df, **arima_params)
results = evaluate_arima(data = df, best_cfg=best_cfg)
print(results)




# import pandas as pd
# import numpy as np
# from models.sarima_model import *

# df = pd.read_csv("/workspaces/benchmark_ts_model_test/univariate/datasets/D1.csv", usecols=[1])
# # Define list of units to search

# sarima_params = {'p_values': [0, 1], 'd_values': [0, 1], 'q_values': [0, 1], 
#                  'P_values': [0, 1], 'D_values': [0, 1], 'Q_values': [0, 1], 'm_values': [12]}
# best_cfg, _ = random_search_sarima(data = df, **sarima_params)
# results = evaluate_sarima(data = df, best_cfg=best_cfg)
# print(results)



