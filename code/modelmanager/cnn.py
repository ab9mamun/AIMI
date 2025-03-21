# -*- coding: utf-8 -*-
from tensorflow.keras import Model
#from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Reshape, Input, Add, Activation
from tensorflow.keras import activations


def create_model_cnn(window_size, feature_size):
    input_frame = Input(shape=(window_size, feature_size, 1))
    x = Conv2D(10, (60, 1), strides=(30,1), activation='relu')(input_frame)
    x = Conv2D(10, (1, 1), strides=(1,1), activation='relu')(x)
    x = Conv2D(10, (1, 1), strides=(1,1), activation='relu')(x)
    x = Conv2D(10, (1, 1), strides=(1,1), activation='relu')(x)
    x = Conv2D(10, (1, 1), strides=(1,1), activation='relu')(x)
    x = Conv2D(10, (1, 1), strides=(1,1), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(1, activation='relu')(x)
    x2 = Reshape((window_size*feature_size*1,))(input_frame)
    x2 = Dense(1)(x2)
    z = Add()([x, x2])
    output = Activation(activations.sigmoid)(z)
    return Model(input_frame, output)