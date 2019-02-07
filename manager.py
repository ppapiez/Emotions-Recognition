import data_reader as dr
import Models.cnn_ke_1
import Models.cnn_yf_1

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

from keras.models import model_from_json
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau


class Manager():
    def __init__(self, args):
        self._path = args.path
        self._output_classes = args.output_classes
        if args.new_model == 'True':
            self._model = self._create_model(args.model_architecture)
            self._history = None
        else:
            self._model = self._load_model()

    def _create_model(self, architecture):
        if architecture == "cnn_yf_1":
            return Models.cnn_yf_1.conv_create()
        elif architecture == "cnn_ke_1":
            return Models.cnn_ke_1.conv_create()

    def _load_model(self):
        with open(os.path.join(self._path, "model.json"), 'r') as json_file:
            loaded_model = model_from_json(json_file.read())

        loaded_model.load_weights(os.path.join(self._path, "model.h5"))

        if os.path.isfile(os.path.join(self._path, "history")):
            with open(os.path.join(self._path, "history"), 'rb') as bin_file:
                self._history = pickle.load(bin_file)
        else:
            self._history = None

        return loaded_model

    def save_model(self):
        """
            Save model architecture, weights and training history.
        """
        model_json = self._model.to_json()
        if not os.path.exists(self._path):
            os.mkdir(self._path)

        with open(os.path.join(self._path, "model.json"), 'w') as json_file:
            json_file.write(model_json)

        self._model.save_weights(os.path.join(self._path, "model.h5"))

        with open(os.path.join(self._path, "history"), 'wb') as bin_file:
            pickle.dump(self._history, bin_file)

    def train_model(self, args):
        """
            Train model.
            Args: dataset, batch_size, epochs.
        """
        X_train, X_test, y_train, y_test = dr.load_data(args.dataset)

        self._model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        self._model.summary()

        learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                                    patience=3,
                                                    verbose=1,
                                                    factor=0.5,
                                                    min_lr=0.000001)

        res = self._model.fit(X_train, y_train, batch_size=args.batch_size, nb_epoch=args.epochs,
                              verbose=1, validation_data=(X_test, y_test),
                              callbacks=[learning_rate_reduction])

        print(self._model.evaluate(X_test, y_test))

        self._update_history(res)
        self.plot_history()

    def classify_image(self, path):
        """ Classify image.
            Returns label and neural network output.
        """
        res = self._model.predict(dr.load_single_image(path))
        dict = dr.get_classes_dict(self._output_classes)
        inv_dict = {v: k for k, v in dict.items()}
        return res, inv_dict[np.argmax(res)]

    def _update_history(self, res):
        if self._history == None:
            self._history = res.history
        else:
            self._history['loss'] += res.history['loss']
            self._history['val_loss'] += res.history['val_loss']
            self._history['acc'] += res.history['acc']
            self._history['val_acc'] += res.history['val_acc']

    def plot_history(self):
        """
            Plot training history: loss and accuracy.
        """
        if self._history != None:
            plt.figure(figsize=(14, 3))
            plt.subplot(1, 2, 1)
            plt.ylabel('Loss', fontsize=16)
            plt.plot(self._history['loss'], color='b', label='Training Loss')
            plt.plot(self._history['val_loss'], color='r', label='Validation Loss')
            plt.legend(loc='upper right')

            plt.subplot(1, 2, 2)
            plt.ylabel('Accuracy', fontsize=16)
            plt.plot(self._history['acc'], color='b', label='Training Accuracy')
            plt.plot(self._history['val_acc'], color='r', label='Validation Accuracy')
            plt.legend(loc='lower right')
            plt.show()
        else:
            print('There is no history to plot!')