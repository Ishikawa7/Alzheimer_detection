import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

def build_Model(input_shape : tuple[int]) -> Model:
    input = layers.Input(shape= input_shape ,name="input")
    flatten = layers.Flatten()(input)
    dense_0 = layers.Dense(300, activation="tanh")(flatten)
    dropout_0 = layers.Dropout(0.3)(dense_0)
    dense_1 = layers.Dense(300, activation="tanh")(dropout_0)
    dropout_1 = layers.Dropout(0.3)(dense_1)
    ##dense_2 = layers.Dense(729, activation="tanh")(dense_1)
    #dropout_2 = layers.Dropout(0.2)(dense_2)
    ##dense_3 = layers.Dense(729, activation="tanh")(dense_2)
    #dropout_3 = layers.Dropout(0.2)(dense_3)
    dense_4 = layers.Dense(200, activation="tanh")(dropout_1)
    dropout_4 = layers.Dropout(0.2)(dense_4)
    dense_5 = layers.Dense(100, activation="tanh", kernel_regularizer='l1')(dropout_4) #
    dropout_5 = layers.Dropout(0.3)(dense_5)

    #output = layers.Dense(1, activation="relu", name="output")(dense_5)
    output_softmax = layers.Dense(2, activation="softmax", kernel_regularizer='l1')(dropout_5)

    model = Model(
        inputs = [input],
        outputs = [output_softmax]
        )

    #model.compile(
    #    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    #    loss=["mse"],
    #    loss_weights=[1.0],
    #    )
    model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model