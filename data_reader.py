import os
import tarfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from skimage import io
import cv2
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


def load_yalefaces(train_size=.8):
    """ Load the Yale Faces data set and generate labels for each image.

        Returns: Train and validation samples with their labels. The training samples are images (243 * 320),
        the labels are one-hot-encoded values for each category
    """
    images_path = [os.path.join(".\Data\Yalefaces", item) for item in os.listdir(".\Data\Yalefaces")]

    image_data = []
    image_labels = []

    for i, im_path in enumerate(images_path):
        im = io.imread(im_path, as_grey=True)
        image_data.append(np.array(im, dtype='uint8'))

        label = get_classes_dict('yalefaces_classes')[os.path.split(im_path)[1].split(".")[1]]
        image_labels.append(label)

    X_ = np.array(image_data).astype(np.float32)
    enc = LabelEncoder()
    y_ = enc.fit_transform(np.array(image_labels))
    y_ = np_utils.to_categorical(y_)
    X_train, X_test, y_train, y_test = train_test_split(X_, y_, train_size=train_size, random_state=22)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))

    return X_train, X_test, y_train, y_test

def load_kaggle_emotions():
    """ Load the Kaggle Emotions data set and generate labels for each image.

        Returns: Train and validation samples with their labels. The training samples are images (48 * 48),
        the labels are one-hot-encoded values for each category
    """
    data_comp = tarfile.open("Data/Kaggle-Emotions/fer2013.tar.gz")
    ds = pd.read_csv(data_comp.extractfile("fer2013/fer2013.csv"))

    train = ds[["emotion", "pixels"]][ds["Usage"] == "Training"]
    train['pixels'] = train['pixels'].apply(lambda x: np.fromstring(x, sep=' '))
    X_train = np.vstack(train['pixels'].values)

    test = ds[["emotion", "pixels"]][ds["Usage"] == "PublicTest"]
    test['pixels'] = test['pixels'].apply(lambda x: np.fromstring(x, sep=' '))
    X_test = np.vstack(test['pixels'].values)

    y_train = np.array(train["emotion"])
    y_test = np.array(test["emotion"])

    X_train = X_train.reshape(-1, 48, 48, 1)
    y_train = np_utils.to_categorical(y_train)
    X_test = X_test.reshape(-1, 48, 48, 1)
    y_test = np_utils.to_categorical(y_test)

    return X_train, X_test, y_train, y_test


def load_data(dataset):
    """ Load chosen data set.

        Returns: Train and validation samples with their labels.
    """
    if dataset == "Yalefaces":
        return load_yalefaces()
    elif dataset == "Kaggle-Emotions":
        return load_kaggle_emotions()

def get_classes_dict(output_classes):
    """
        Returns: dictionary with chosen data set categories.
    """
    if output_classes == "yalefaces_classes":
        return {'centerlight': 0, 'glasses': 1, 'happy': 2, 'leftlight': 3, 'noglasses': 4, 'normal': 5, 'rightlight': 6,
         'sad': 7, 'sleepy': 8, 'surprised': 9, 'wink': 10}
    elif output_classes == "kaggle_classes":
        return {'anger': 0, 'disgust': 1, 'fear': 2, 'happiness': 3, 'neutral': 4, 'sadness': 5, 'surprise': 6}

def load_single_image(path):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return im.reshape(1, im.shape[0], im.shape[1], 1)


# if __name__ == '__main__':
#     X_train, X_test, y_train, y_test = kaggle_emotions_load()
# #     X_train, X_test, y_train, y_test = yalebase_load()
#
#     plt.imshow(X_train[5].reshape(48, 48), cmap='gray')
#     plt.show()
#     print(y_train[5])