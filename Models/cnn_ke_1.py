from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPool2D, Dropout, Dense, BatchNormalization, Flatten

def conv_create():
    """
        Returns CNN model for input images (48, 48). Compatible with Kaggle-Emotions data set.
    """
    model = Sequential()
    model.name = "cnn_ke_1"
    model.add(Conv2D(64, 5, data_format="channels_last", kernel_initializer="he_normal",
                     input_shape=(48, 48, 1)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv2D(64, 4))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, 3))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv2D(128, 3))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(Dense(7))
    model.add(Activation('softmax'))

    return model