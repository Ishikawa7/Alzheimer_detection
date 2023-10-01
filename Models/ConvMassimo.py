import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, AveragePooling2D, Dropout

def build_Model(input_shape : tuple[int]) -> Sequential:
    model = Sequential()
    model.model_name = "ConvMassimo"
    model.add(Conv2D(filters = 16, kernel_size=4, activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters = 32, kernel_size=4, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters = 64, kernel_size=4, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(200, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation="relu"))
    model.add(Dropout(0.2))
    #model.add(Dense(1, activation="relu"))
    model.add(Dense(4, activation="softmax"))
    #model.compile(optimizer="adam", loss="mse")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

#   model = Sequential()
#   #add model layers - RP Architettura
#   model.add(Conv2D(16, kernel_size=1, activation="relu", input_shape=(img_height, img_width, num_channels)))
#   model.add(Conv2D(16, kernel_size=3, activation="relu"))
#   model.add(MaxPooling2D(pool_size=(2, 2)))
#   model.add(Conv2D(16, kernel_size=3, activation="relu")) #3
#   model.add(Conv2D(32, kernel_size=3, activation="relu")) #3
#   model.add(MaxPooling2D(pool_size=(2, 2)))
#   model.add(Conv2D(32, kernel_size=3, activation="relu")) #3
#   model.add(Conv2D(64, kernel_size=3, activation="relu")) #3
#   model.add(MaxPooling2D(pool_size=(2, 2)))
#   model.add(Conv2D(64, kernel_size=3, activation="relu")) #3
#   model.add(Conv2D((96), kernel_size=3, activation="relu")) #3
#   model.add(MaxPooling2D(pool_size=(2, 2)))
#   #model.add(Conv2D(64, kernel_size=3, activation="relu")) #3
#   #model.add(Conv2D(128, kernel_size=3, activation="relu")) #3
#   #model.add(MaxPooling2D(pool_size=(2, 2)))
#   #model.add(Conv2D(15, kernel_size=3,  padding = 'SAME', activation="relu")) #3
#   # model.add(MaxPooling2D(pool_size=(2, 2)))
#   # model.add(Conv2D(10, kernel_size=3, activation="relu")) #3
#   # model.add(MaxPooling2D(pool_size=(2, 2)))
#   # model.add(Conv2D(15, kernel_size=3, activation="relu"))
#   model.add(Flatten())
#   model.add(Dense(200, activation="relu"))
#   model.add(Dropout(0.4))
#   model.add(Dense(100, activation="relu"))
#   model.add(Dropout(0.4))
#   model.add(Dense(Class_Count, activation="softmax"))