from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model

def buid_Model(img_height, img_width):
    input = layers.Input(shape=(img_height, img_width, 1))

    # Encoder
    conv_1 = layers.Conv2D(filters = 32, kernel_size=3, activation="relu", padding="same",input_shape=(img_height, img_width, 1))(input)
    poolin_1 = layers.MaxPooling2D((2, 2), padding="same")(conv_1)
    conv_2 = layers.Conv2D(filters = 32, kernel_size=3, activation="relu", padding="same")(poolin_1)
    latent_space = layers.MaxPooling2D((2, 2), padding="same")(conv_2)

    # Decoder
    conv_T1 = layers.Conv2DTranspose(filters = 32, kernel_size=3, strides=2, activation="relu", padding="same")(latent_space)
    conv_T2 = layers.Conv2DTranspose(filters = 32, kernel_size=3, strides=2, activation="relu", padding="same")(conv_T1)
    output = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(conv_T2)

    # Autoencoder
    autoencoder = Model(
        inputs = [input],
        outputs = [output, latent_space]
        )
    autoencoder.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        loss_weights=[1.0, 0.0],
        )
    return autoencoder
