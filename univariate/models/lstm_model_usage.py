# import numpy as np
# import pandas as pd
# from keras.models import Sequential
# from keras.layers import LSTM, Dense
# from sklearn.preprocessing import MinMaxScaler
# from sklearn import preprocessing
# import math, time
# from sklearn.metrics import mean_squared_error




# def create_dataset(dataset, look_back=15):
#     dataX, dataY = [], []
#     for i in range(len(dataset)-look_back-1):
#         a = dataset[i:(i+look_back), 0]
#         dataX.append(a)
#         dataY.append(dataset[i + look_back, 0])
#     return np.array(dataX), np.array(dataY)


# def build_lstm():
#     look_back = 15
#     model = Sequential()
#     # model.add(LSTM(20, input_shape=(1, look_back)))
#     model.add(LSTM(20, input_shape=(look_back, 1)))

#     model.add(Dense(1))
#     model.compile(loss='mean_squared_error', optimizer='adam')

#     return model


# def evaluate_lstm(model, train, test):

#     x_train, y_train = create_dataset(train, look_back=15)
#     x_test, y_test = create_dataset(test, look_back=15)
#     model.fit(x_train, y_train, epochs=3, batch_size=1, verbose=2)


#     trainPredict = model.predict(x_train)
#     testPredict = model.predict(x_test)
#     # invert predictions
#     trainPredict = min_max_scaler.inverse_transform(trainPredict)
#     trainY = min_max_scaler.inverse_transform([y_train])
#     testPredict = min_max_scaler.inverse_transform(testPredict)
#     testY = min_max_scaler.inverse_transform([y_test])

#     # calculate root mean squared error
#     trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
#     print('Train Score: %.2f RMSE' % (trainScore))
#     testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
#     print('Test Score: %.2f RMSE' % (testScore))

#     return testScore



# datapath = "datasets/candy_production.csv"
# df = pd.read_csv(datapath, index_col = 0, parse_dates=True)

# print(df.head())

# min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
# dataset = min_max_scaler.fit_transform(df.values.reshape(-1, 1))

# # split into train and test sets
# train_size = int(len(dataset) * 0.8)
# test_size = len(dataset) - train_size
# train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]


# lstm_model = build_lstm()
# mse = evaluate_lstm(model=lstm_model, train=train, test=test)

# print("cat")


# import numpy as np
# import pandas as pd
# from keras.models import Sequential
# from keras.layers import LSTM, Dense
# from sklearn.preprocessing import MinMaxScaler
# from sklearn import preprocessing
# import math, time
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# import itertools


# def create_dataset(dataset, look_back=15):
#     dataX, dataY = [], []
#     for i in range(len(dataset)-look_back-1):
#         a = dataset[i:(i+look_back), 0]
#         dataX.append(a)
#         dataY.append(dataset[i + look_back, 0])
#     return np.array(dataX), np.array(dataY)


# def build_lstm(units, look_back = 15):
#     model = Sequential()
#     # model.add(LSTM(20, input_shape=(1, look_back)))
#     model.add(LSTM(units, input_shape=(look_back, 1)))

#     model.add(Dense(1))
#     model.compile(loss='mean_squared_error', optimizer='adam')

#     return model


# def evaluate_lstm(model, train, test, epochs):

#     x_train, y_train = create_dataset(train, look_back=15)
#     x_test, y_test = create_dataset(test, look_back=15)

#     model.fit(x_train, y_train, epochs=epochs, batch_size=1, verbose=2)


#     trainPredict = model.predict(x_train)
#     testPredict = model.predict(x_test)

#     # invert predictions
#     trainPredict = min_max_scaler.inverse_transform(trainPredict)
#     trainY = min_max_scaler.inverse_transform([y_train])
#     testPredict = min_max_scaler.inverse_transform(testPredict)
#     testY = min_max_scaler.inverse_transform([y_test])

  
#     # calculate mape
#     trainScore =  np.mean(np.abs((trainY[0] - trainPredict[:, 0]) / trainY[0])) * 100
#     testScore = np.mean(np.abs((testY[0] - testPredict[:, 0]) / testY[0])) * 100

#     print('Train Score: %.2f mape' % (trainScore))
#     print('Test Score: %.2f mape' % (testScore))


#     return testScore

# def evaluate_final_model(model, train, test, epochs):

#     x_train, y_train = create_dataset(train, look_back=15)
#     x_test, y_test = create_dataset(test, look_back=15)

#     model.fit(x_train, y_train, epochs=epochs, batch_size=1, verbose=2)


#     testPredict = model.predict(x_test)

#     # invert predictions
#     testPredict = min_max_scaler.inverse_transform(testPredict)
#     testY = min_max_scaler.inverse_transform([y_test])

  
#     # calculate mape
#     mse = mean_squared_error(testY[0], testPredict[:, 0])
#     mae = mean_absolute_error(testY[0], testPredict[:, 0])
#     rmse = math.sqrt(mse)
#     mape = np.mean(np.abs((testY[0] - testPredict[:, 0]) / testY[0])) * 100

#     return mae, mape, mse, rmse


# def perform_grid_search(df, units_list, epochs_list):
#     # Preprocess the data
#     min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
#     dataset = min_max_scaler.fit_transform(df.values.reshape(-1, 1))


#     # Split into train and test sets
#     train_size = int(len(dataset) * 0.8)
#     test_size = len(dataset) - train_size
#     train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

#     # Variables to track the best model
#     best_mape = float('inf')
#     best_units = None
#     best_epochs = None


#     # Perform grid search
#     for units, epochs in itertools.product(units_list, epochs_list):
#         print(f"Evaluating model with {units} units and {epochs} epochs...")
#         lstm_model = build_lstm(units)
#         mape = evaluate_lstm(lstm_model, train, test, epochs)


#         # Update the best model if this one is better
#         if mape < best_mape:
#             best_mape = mape
#             best_units = units
#             best_epochs = epochs

#     # evaluate the final model
#     final_model = build_lstm(units=units)
#     mae, mape, mse, rmse = evaluate_final_model(final_model, train=train, test=test, epochs=epochs)
#     return mae, mape, mse, rmse


# # Load the dataset
# datapath = "datasets/candy_production.csv"
# df = pd.read_csv(datapath, index_col=0, parse_dates=True)

# # # Define grid search parameters
# # units_list = [10, 20, 50]  # Different LSTM units to try
# # epochs_list = [3, 5, 10]   # Different epoch values to try

# # Define grid search parameters
# units_list = [10, 20]  # Different LSTM units to try
# epochs_list = [2, 3]   # Different epoch values to try


# # Perform grid search
# mae, mape, mse, rmse = perform_grid_search(df, units_list, epochs_list)
# print(f"mae : {mae}")
# print(f"mae : {mape}")

# print(f"mae : {mse}")
# print(f"mae : {rmse}")


import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
import itertools

def create_dataset(dataset, look_back=15):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def build_lstm(units, look_back=15):
    model = Sequential()
    model.add(LSTM(units, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def evaluate_lstm(model, train, test, epochs, scaler):
    x_train, y_train = create_dataset(train, look_back=15)
    x_test, y_test = create_dataset(test, look_back=15)

    # Reshape the input to [samples, timesteps, features]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    model.fit(x_train, y_train, epochs=epochs, batch_size=1, verbose=2)

    # Make predictions
    trainPredict = model.predict(x_train)
    testPredict = model.predict(x_test)

    # Invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([y_train])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([y_test])

    # Calculate MAPE
    trainScore = np.mean(np.abs((trainY[0] - trainPredict[:, 0]) / trainY[0])) * 100
    testScore = np.mean(np.abs((testY[0] - testPredict[:, 0]) / testY[0])) * 100

    print('Train Score: %.2f MAPE' % (trainScore))
    print('Test Score: %.2f MAPE' % (testScore))

    return testScore

def evaluate_final_model(model, train, test, epochs, scaler):
    x_train, y_train = create_dataset(train, look_back=15)
    x_test, y_test = create_dataset(test, look_back=15)

    # Reshape the input to [samples, timesteps, features]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    model.fit(x_train, y_train, epochs=epochs, batch_size=1, verbose=2)

    # Make predictions
    testPredict = model.predict(x_test)

    # Invert predictions
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([y_test])

    # Calculate metrics
    mse = mean_squared_error(testY[0], testPredict[:, 0])
    mae = mean_absolute_error(testY[0], testPredict[:, 0])
    rmse = math.sqrt(mse)
    mape = np.mean(np.abs((testY[0] - testPredict[:, 0]) / testY[0])) * 100

    return mae, mape, mse, rmse

def perform_grid_search(df, units_list, epochs_list):
    # Preprocess the data
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = min_max_scaler.fit_transform(df.values.reshape(-1, 1))

    # Split into train and test sets
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    # Variables to track the best model
    best_mape = float('inf')
    best_units = None
    best_epochs = None

    # Perform grid search
    for units, epochs in itertools.product(units_list, epochs_list):
        print(f"Evaluating model with {units} units and {epochs} epochs...")
        lstm_model = build_lstm(units)
        mape = evaluate_lstm(lstm_model, train, test, epochs, min_max_scaler)

        # Update the best model if this one is better
        if mape < best_mape:
            best_mape = mape
            best_units = units
            best_epochs = epochs

    # Evaluate the final model
    final_model = build_lstm(units=best_units)
    mae, mape, mse, rmse = evaluate_final_model(final_model, train, test, epochs=best_epochs, scaler=min_max_scaler)
    return mae, mape, mse, rmse

# Load the dataset
datapath = "datasets/candy_production.csv"
df = pd.read_csv(datapath, index_col=0, parse_dates=True)

# Define grid search parameters
units_list = [10, 20]  # Different LSTM units to try
epochs_list = [2, 3]   # Different epoch values to try

# Perform grid search
mae, mape, mse, rmse = perform_grid_search(df, units_list, epochs_list)
print(f"MAE: {mae}")
print(f"MAPE: {mape}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
