import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

def build_Model(img_height, img_width):
    input = layers.Input(shape=(img_height, img_width, 1),name="input")
    flatten = layers.Flatten()(input)
    # Encoder
    #dense_0 = layers.Dense(1000, activation="tanh")(flatten)
    #dense_1 = layers.Dense(500, activation="tanh")(dense_0)
    dense_2 = layers.Dense(250, activation="tanh")(flatten)
    latent_space = layers.Dense(100, activation="tanh", kernel_regularizer='l1')(dense_2)

    encoder_model = Model(inputs=[input], outputs=[latent_space])

    # Decoder
    dense_3 = layers.Dense(250, activation="tanh")(latent_space)
    #dense_4 = layers.Dense(500, activation="tanh")(dense_3)
    dense_5 = layers.Dense(img_height * img_width, activation="tanh")(dense_3)
    output = layers.Reshape((img_height, img_width, 1),name="output_autoencoder")(dense_5)

    # Autoencoder
    autoencoder = Model(
        inputs = [input],
        outputs = [output]
        )

    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=["mse"],
        loss_weights=[1.0],
        )
    return autoencoder, encoder_model