import pandas as pd
import numpy as np
import math

from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#Read data from given url and extract the second column
def read_data(url):
    df = pd.read_csv(url, usecols=[1], engine='python')
    data = np.array(df.values.astype('float32'))
#Normalise data into (0,1) range 
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data).flatten()
    n = len(data)
    return data, n

sunspots_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv'
data, n = read_data(sunspots_url)

print(data)

#Splitting data into train and test based on split ratio
def get_train_test(split_percent, data):
    n = len(data)
    split = int(n * split_percent)
    train_data = data[:split]
    test_data = data[split:]
    return train_data, test_data

split_percent = 0.8
train_data, test_data = get_train_test(split_percent, data)


#Reshape data into input-output pairs with specified time steps
def get_XY(dat, time_steps):
    Y_ind = np.arange(time_steps, len(dat), time_steps)
    Y = dat[Y_ind]
    rows_x = len(Y)
#Prepare Training and testing data
    X = dat[range(time_steps*rows_x)]
    X = np.reshape(X, (rows_x, time_steps, 1))    
    return X, Y


time_steps = 12
trainX, trainY = get_XY(train_data, time_steps)
testX, testY = get_XY(test_data, time_steps)


#Define the RNN model
def create_RNN(units, dense_units, input_shape, activation):
    model = Sequential()
    model.add(SimpleRNN(units, input_shape=input_shape, 
                        activation=activation[0], return_sequences=True))
    model.add(Dense(dense_units, activation=activation[1]))
#Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

model = create_RNN(units=3, dense_units=1, input_shape=(time_steps,1), 
                   activation=['tanh', 'tanh'])
model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2)

#Get error of predictions to evaluate it
def print_error(trainY, testY, train_predict, test_predict): 
    train_predict = train_predict.reshape(-1)
    test_predict = test_predict.reshape(-1)
    train_rmse = math.sqrt(mean_squared_error(trainY,train_predict))
    test_rmse = math.sqrt(mean_squared_error(testY, test_predict))
    print('Train RMSE: %.3f RMSE' % (train_rmse))
    print('Test RMSE: %.3f RMSE' % (test_rmse)) 