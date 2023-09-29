import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, AveragePooling2D, Dropout
from tensorflow.keras.models import model_from_json 
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def build_Model(img_height, img_width):
    model = Sequential()
    model.model_name = "ConvMassimo"
    model.add(Conv2D(filters = 32, kernel_size=4, activation="relu", input_shape=(img_height, img_width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters = 32, kernel_size=3, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters = 32, kernel_size=2, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(200, activation="gelu"))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation="gelu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="gelu"))
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model