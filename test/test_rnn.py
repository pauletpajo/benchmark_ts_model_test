
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')


import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
from sklearn.model_selection import train_test_split

from models.rnn_model import *




data = pd.read_csv('/workspaces/benchmark_ts_model_test/datasets/candy_production.csv', index_col = 0, parse_dates = True)

# Splitting data into training and testing sets (80% training, 20% testing)
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)



print(data)
# Train and evaluate RNN
rnn_model = train_rnn(train_data)
rnn_mae = evaluate_rnn(rnn_model, test_data)

print(rnn_mae)
