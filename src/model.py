import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization
from tensorflow.keras.layers import Bidirectional, LSTM
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Reshape, Activation

def build_model():

    model = Sequential()

    # Input shape = (76,1)
    model.add(
        Conv1D(
            filters=64,
            kernel_size=76,
            padding="same",
            activation="relu",
            input_shape=(76,1)
        )
    )

    model.add(BatchNormalization())

    model.add(
        Bidirectional(
            LSTM(64, return_sequences=False)
        )
    )

    model.add(Reshape((128,1)))

    model.add(BatchNormalization())

    model.add(
        Bidirectional(
            LSTM(128, return_sequences=False)
        )
    )

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(3))
    model.add(Activation('softmax'))

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    return model