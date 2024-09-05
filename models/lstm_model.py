import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense


def train_lstm(data):
    model = Sequential()
    model.add(LSTM(50, input_shape=(None, 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    data = np.array(data).reshape(-1, 1)
    model.fit(data[:-1], data[1:], epochs=10, verbose=0)

    return model


def evaluate_lstm(model, data):
    data = np.array(data).reshape(-1, 1)
    predictions = model.predict(data[:-1])
    mae = np.mean(abs(predictions.flatten() - data[1:].flatten()))
    return mae
