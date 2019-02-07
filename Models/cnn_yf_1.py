from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPool2D, Dropout, Dense, BatchNormalization, Flatten

def conv_create():
    """
        Returns CNN model for input images (243, 320). Compatible with Yale Faces data set.
    """
    model = Sequential()
    model.name = "cnn_yf_1"
    model.add(Conv2D(50, 5, data_format="channels_last", kernel_initializer="he_normal",
                     input_shape=(243, 320, 1)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv2D(50, 4))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(3, 3), strides=3))
    model.add(Dropout(0.5))

    model.add(Conv2D(100, 3))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv2D(100, 3))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(MaxPool2D(pool_size=(5, 5), strides=5))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(Dense(11))
    model.add(Activation('softmax'))

    return model