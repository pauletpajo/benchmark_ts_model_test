# import numpy as np
# from keras.models import Sequential
# from keras.layers import SimpleRNN, Dense


# def train_rnn(data):
#     model = Sequential()
#     model.add(SimpleRNN(50, input_shape=(None, 1)))
#     model.add(Dense(1))
#     model.compile(optimizer="adam", loss="mse")

#     data = np.array(data).reshape(-1, 1)
#     model.fit(data[:-1], data[1:], epochs=10, verbose=0)

#     return model


# def evaluate_rnn(model, data):
#     data = np.array(data).reshape(-1, 1)
#     predictions = model.predict(data[:-1])
#     mae = np.mean(abs(predictions.flatten() - data[1:].flatten()))
#     return mae

import numpy as np
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense

def train_rnn(train_data):
    model = Sequential()
    model.add(SimpleRNN(50, input_shape=(None, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    train_data = np.array(train_data).reshape(-1, 1)
    model.fit(train_data[:-1], train_data[1:], epochs=10, verbose=0)
    
    return model

def evaluate_rnn(model, test_data):
    test_data = np.array(test_data).reshape(-1, 1)
    predictions = model.predict(test_data[:-1])
    mae = np.mean(abs(predictions.flatten() - test_data[1:].flatten()))
    return mae

