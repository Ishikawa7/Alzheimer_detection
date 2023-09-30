import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

def build_Model(input_shape : tuple[int]) -> Model:
    input = layers.Input(shape= input_shape ,name="input")
    flatten = layers.Flatten()(input)
    dense_0 = layers.Dense(300, activation="tanh")(flatten)
    #dense_1 = layers.Dense(300, activation="relu")(dense_0)
    #dense_2 = layers.Dense(300, activation="relu")(dense_1)
    #dense_3 = layers.Dense(300, activation="relu")(dense_2)
    #dense_4 = layers.Dense(300, activation="relu")(dense_3)
    dense_5 = layers.Dense(300, activation="tanh")(dense_0)
    output = layers.Dense(1, activation="tanh", name="output_class_regression")(dense_5)

    # Autoencoder
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