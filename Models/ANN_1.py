import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

def build_Model(input_shape : tuple[int]) -> Model:
    input = layers.Input(shape= input_shape ,name="input")
    flatten = layers.Flatten()(input)
    dense_0 = layers.Dense(500, activation="gelu")(flatten)
    #dropout_0 = layers.Dropout(0.2)(dense_0)
    dense_1 = layers.Dense(500, activation="gelu")(dense_0)
    #dropout_1 = layers.Dropout(0.2)(dense_1)
    dense_2 = layers.Dense(500, activation="gelu")(dense_1)
    #dropout_2 = layers.Dropout(0.2)(dense_2)
    dense_3 = layers.Dense(500, activation="gelu")(dense_2)
    #dropout_3 = layers.Dropout(0.2)(dense_3)
    dense_4 = layers.Dense(250, activation="gelu")(dense_3)
    #dropout_4 = layers.Dropout(0.2)(dense_4)
    dense_5 = layers.Dense(100, activation="gelu")(dense_4)
    #dropout_5 = layers.Dropout(0.2)(dense_5)
    output = layers.Dense(1, activation="gelu", name="output")(dense_5)

    model = Model(
        inputs = [input],
        outputs = [output]
        )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=["mse"],
        loss_weights=[1.0],
        )
    return model