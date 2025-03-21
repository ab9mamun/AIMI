# -*- coding: utf-8 -*-
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation, InputLayer
from tensorflow.keras import activations

def create_model_lstm(window_size, feature_size):
    batch_size =1
    model = Sequential()
    #input_frame = Input(shape=(window_size, feature_size))
    model.add(InputLayer(input_shape=(window_size, feature_size)))
    model.add(LSTM(2))#, return_sequences=True))
    #model.add(LSTM(1, batch_input_shape=(batch_size, window_size, feature_size)))
    model.add(Dense(1))
    model.add(Activation(activations.sigmoid))
    return model