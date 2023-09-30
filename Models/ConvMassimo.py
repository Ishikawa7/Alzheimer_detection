import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, AveragePooling2D, Dropout

def build_Model(input_shape : tuple[int]) -> Sequential:
    model = Sequential()
    model.model_name = "ConvMassimo"
    model.add(Conv2D(filters = 16, kernel_size=7, activation="gelu", input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters = 32, kernel_size=7, activation="gelu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(200, activation="gelu"))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation="gelu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="relu"))
    model.compile(optimizer="adam", loss="mse")
    return model