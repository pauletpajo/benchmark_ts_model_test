import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import math, time
from sklearn.metrics import mean_squared_error




def create_dataset(dataset, look_back=15):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


def build_lstm():
    look_back = 15
    model = Sequential()
    model.add(LSTM(20, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model


def evaluate_lstm(model, train, test):

    x_train, y_train = create_dataset(train, look_back=15)
    x_test, y_test = create_dataset(test, look_back=15)
    model.fit(x_train, y_train, epochs=20, batch_size=1, verbose=2)


    trainPredict = model.predict(x_train)
    testPredict = model.predict(x_test)
    # invert predictions
    trainPredict = min_max_scaler.inverse_transform(trainPredict)
    trainY = min_max_scaler.inverse_transform([y_train])
    testPredict = min_max_scaler.inverse_transform(testPredict)
    testY = min_max_scaler.inverse_transform([y_test])

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))

    return testScore




df = pd.read_csv('test\data\candy_production.csv', index_col = 0, parse_dates=True)

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
dataset = min_max_scaler.fit_transform(df.values.reshape(-1, 1))

# split into train and test sets
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]


lstm_model = build_lstm()
mse = evaluate_lstm(model=lstm_model, train=train, test=test)