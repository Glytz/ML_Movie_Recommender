from __future__ import print_function, division

import os
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
import data_parser
from data_parser import get_formated_data

#implementation of a simple MLP to try and predict vote with user content
class MLP:
    def __init__(self, load_model):
        self.number_of_movie_genre = 0
        self.number_of_occupations = 0
        # we calculate the shape : movie_genre + occupations + adress code + sex + age
        #self.input_shape = (self.movie_genres + self.number_of_movie_genre + 1 + 1 + 1)
        #temp values to test on mnist
        self.input_shape = (43,)
        self.num_classes = 5
        if not load_model:
            optimizer = Adam(0.0002, 0.5)

            # Build and compile the discriminator
            self.classifier = self.build_classifier()
            self.classifier.compile(loss='categorical_crossentropy',
                                    optimizer=optimizer,
                                    metrics=['acc'])
        else:
            working_directory_path = os.getcwd()
            self.classifier = keras.models.load_model(working_directory_path + "/models/classifier.h5")

    def build_classifier(self):
        model = Sequential()

        #model.add(Flatten(input_shape=self.input_shape))
        model.add(Dense(512, input_shape=self.input_shape))
        model.add((LeakyReLU(alpha=0.2)))
        model.add(Dropout(0.5))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.summary()

        img = Input(shape=self.input_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs=1, batch_size=128):

        # load the dataset
        # dummy dataset
        dataset = get_formated_data()
        # seperate data into training, validation and test
        training_set = dataset[0:int(np.floor(dataset.shape[0] * 0.7))]
        x_train = training_set[:, 0]
        y_train = training_set[:, 1]

        validation_set = dataset[int(np.floor(dataset.shape[0] * 0.7)) + 1:int(np.floor(dataset.shape[0] * 0.85))]
        x_valid = validation_set[:, 0]
        y_valid = validation_set[:, 1]

        test_set = dataset[int(np.floor(dataset.shape[0] * 0.85)) + 1:int(np.floor(dataset.shape[0]) - 1)]
        x_test = test_set[:, 0]
        y_test = test_set[:, 1]

        #=============================
        #test mnist data
        #(x_train, y_train), (x_valid, y_valid) = mnist.load_data()
        #x_test = x_valid
        #y_test = y_valid
        #=============================
        #convert training data to one hot
        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_valid = keras.utils.to_categorical(y_test, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        # callbacks
        save_model = SaveModel(self.classifier)
        history = History()
        # train the model
        training_history = self.classifier.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2,
                                               callbacks=[save_model, history],
                                               validation_split=0.0, validation_data=(x_valid, y_valid),
                                               shuffle=True)
        # test the model on the best model
        # load the best model
        working_directory_path = os.getcwd()
        self.classifier = keras.models.load_model(working_directory_path + "/models/classifier.h5")
        test_metrics = self.classifier.evaluate(x=x_test, y=y_test, batch_size=batch_size, verbose=1)
        print("Test Accuracy : " + str(test_metrics[1]))
        print("Test loss : " + str(test_metrics[0]))

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
    def __init__(self, classifier):
        self.classifier = classifier
    def on_train_begin(self, logs={}):
        self.max_accuracy = 0

    def on_epoch_end(self, epoch, logs={}):
        current_accuracy = logs.get('val_acc')
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
