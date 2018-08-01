import keras
from keras.models import Sequential
from keras.layers import Conv2D


def CNN(input_shape):
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same',
                     kernel_initializer='he_normal', input_shape=input_shape,
                     data_format='channels_first')
             )
    model.add(Conv2D(8, kernel_size=(1, 1), activation='relu', padding='same',
                     kernel_initializer='he_normal', data_format='channels_first')
             )
    model.add(Conv2D(1, kernel_size=(1, 1), activation='sigmoid',
                     kernel_initializer='he_normal', data_format='channels_first')
             )
    return model