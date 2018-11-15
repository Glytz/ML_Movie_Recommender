from __future__ import print_function, division

import os
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam


#implementation of a simple MLP to try and predict vote with user content
class MLP:
    def __init__(self, load_model):
        self.number_of_movie_genre = 0
        self.number_of_occupations = 0
        # we calculate the shape : movie_genre + occupations + adress code + sex + age
        self.input_shape = (self.movie_genres + self.number_of_movie_genre + 1 + 1 + 1)
        if not load_model:
            optimizer = Adam(0.0002, 0.5)

            # Build and compile the discriminator
            self.classifier = self.build_classifier()
            self.classifier.compile(loss='categorical_crossentropy',
                                    optimizer=optimizer,
                                    metrics=['acc'])
        else:
            working_directory_path = os.getcwd()
            self.classifier = keras.models.load_model(working_directory_path + "/classifier" + ".h5")

    def build_classifier(self):
        model = Sequential()

        model.add(Flatten(input_shape=self.input_shape))
        model.add(Dense(512))
        model.add((LeakyReLU(alpha=0.2)))
        model.add(Dropout(0.5))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))
        model.add(Dense(5, activation='softmax'))

        model.summary()

        img = Input(shape=self.input_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # load the dataset
        # dummy dataset
        dataset = np.zeros([[1], [1]])
        # seperate data into training, validation and test
        training_set = dataset[0:np.floor(dataset.shape[0] * 0.7)]
        train_x = training_set[:, 0]
        train_y = training_set[:, 1]

        validation_set = dataset[np.floor(dataset.shape[0] * 0.7) + 1:np.floor(dataset.shape[0] * 0.85)]
        validation_x = validation_set[:, 0]
        validation_y = validation_set[:, 1]

        test_set = dataset[np.floor(dataset.shape[0] * 0.85) + 1:np.floor(dataset.shape[0]) - 1]
        test_x = test_set[:, 0]
        test_y = test_set[:, 1]

        # callbacks
        save_model = SaveModel()
        history = History()
        # train the model
        training_history = self.classifier.fit(train_x, train_y, batch_size=batch_size, epochs=1, verbose=2,
                                               callbacks=[save_model, history],
                                               validation_split=0.0, validation_data=(validation_x, validation_y),
                                               shuffle=True)
        # test the model
        test_metrics = self.classifier.evaluate(x=test_x, y=test_y, batch_size=batch_size, verbose=1,
                                                sample_weight=None, steps=None)

        # make graph for the loss and accuracy at each epoch for validation and traing acc
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.plot(history.accuracies, label="Training Accuracy")
        plt.plot(history.val_accuracies, label="Validation Accuracy")
        plt.title("Accuracy over epochs")
        plt.legend()
        plt.show()

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(history.losses, label="Training Loss")
        plt.plot(history.val_losses, label="Validation Loss")
        plt.title("Loss over epochs")
        plt.legend()
        plt.show()


class SaveModel(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.max_accuracy = 0

    def on_epoch_end(self, epoch, logs={}):
        current_accuracy = logs.get('accuracy')
        if current_accuracy > self.max_accuracy:
            self.max_accuracy = current_accuracy
            # save only the 'best' model
            directory_path = os.getcwd() + "/models"
            if not os.path.exists(directory_path):
                os.mkdir(directory_path)
            prefix = "models/"
            suffix = ".h5"
            self.classifier.save(prefix + "classifier" + suffix)  # save the generator model


class History(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracies = []
        self.val_accuracies = []
        self.val_losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('acc'))
        self.val_accuracies.append(logs.get('val_acc'))
        self.val_losses.append(logs.get('val_loss'))
