from __future__ import print_function, division

import os
import keras
import matplotlib.pyplot as plt
import numpy as np
import utils
from keras.datasets import mnist
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, Softmax
from keras.models import Sequential, Model
from keras.optimizers import Adam
import data_parser
from data_parser import get_formated_data

#implementation of a simple MLP to try and predict vote with user content
from data_parser import get_X, get_Y, get_formated_data2


class MLP:
    def __init__(self, use_data_v1, load_model, model_path):
        self.use_data_v1 = use_data_v1
        if use_data_v1:
            self.number_of_user_info = 23
            self.number_of_item_info = 19
            # we calculate the shape : movie_genre + occupations + adress code + sex + age
            #self.input_shape = (self.movie_genres + self.number_of_movie_genre + 1 + 1 + 1)
            #temp values to test on mnist
            self.input_shape = (self.number_of_user_info + self.number_of_item_info,)
            self.num_classes = 5
        else: # we use the second dataset which will have different input size, and different content
            self.number_of_user_info = 23
            self.number_of_item_info = 19
            self.number_of_movie = 1663
            # we calculate the shape : movie_genre + occupations + adress code + sex + age
            #self.input_shape = (self.movie_genres + self.number_of_movie_genre + 1 + 1 + 1)
            #temp values to test on mnist
            self.input_shape = (self.number_of_user_info + self.number_of_item_info + self.number_of_movie,)
            self.num_classes = 5
        if not load_model:
            optimizer = Adam(0.0002, 0.5)
            #optimizer = Adam(0.1, 0.5)

            # Build and compile the discriminator
            self.classifier = self.build_classifier()
            self.classifier.compile(loss='categorical_crossentropy',
                                    optimizer=optimizer,
                                    metrics=['acc'])
        else:
            working_directory_path = os.getcwd()
            self.classifier = keras.models.load_model(working_directory_path + model_path)

    def build_classifier(self):
        model = Sequential()

        #model.add(Flatten(input_shape=self.input_shape))

        model.add(Dense(1024, input_shape=self.input_shape))
        model.add(BatchNormalization())
        model.add((LeakyReLU(alpha=0.2)))
        model.add(Dropout(0.5))

        model.add(Dense(512))
        model.add(BatchNormalization())
        model.add((LeakyReLU(alpha=0.2)))
        model.add(Dropout(0.5))

        model.add(Dense(256))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))

        model.add(Dense(128))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))

        model.add(Dense(self.num_classes, activation='softmax'))
        #model.add(Dense(self.num_classes, activation='sigmoid'))
       #model.add(BatchNormalization())
       #model.add(keras.activations.sigmoid(1.0))

        model.summary()

        img = Input(shape=self.input_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self,epochs=1, batch_size=128):

        # load the dataset
        # dummy dataset
        if self.use_data_v1:
            dataset = get_formated_data()
        else:
            dataset = get_formated_data2()
        np.random.shuffle(dataset)
        x = get_X(dataset)
        y = get_Y(dataset)
        y[:] -= 1 #we need to seperate into categories, we will have to add  1 to each votes made by the model afterward
        #y[:] /= 4
        # seperate data into training, validation and test
        x_train = x[0:int(np.floor(x.shape[0] * 0.7))]
        y_train = y[0:int(np.floor(y.shape[0] * 0.7))]

        x_valid = x[int(np.floor(x.shape[0] * 0.7)) + 1:int(np.floor(x.shape[0] * 0.85))]
        y_valid = y[int(np.floor(y.shape[0] * 0.7)) + 1:int(np.floor(y.shape[0] * 0.85))]

        x_test = x[int(np.floor(x.shape[0] * 0.85)) + 1:int(np.floor(x.shape[0]) - 1)]
        y_test = y[int(np.floor(y.shape[0] * 0.85)) + 1:int(np.floor(y.shape[0]) - 1)]

        #=============================
        #test mnist data
        #(x_train, y_train), (x_valid, y_valid) = mnist.load_data()
        #x_test = x_valid
        #y_test = y_valid
        #=============================
        #convert training data to one hot
        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_valid = keras.utils.to_categorical(y_valid, self.num_classes)
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

    def softmax_pred_to_list(self, softmax_predictions):
        predict_class = np.argmax(softmax_predictions, axis=1)
        #preds are between 0 and 4, we shift to be between 1 and 5
        predict_class[:] += 1
        return predict_class.tolist()


    def test_model(self, is_soft_max):
        # load the dataset
        # dummy dataset
        dataset = []
        if self.use_data_v1:
            dataset = get_formated_data()
        else:
            dataset = get_formated_data2()
        np.random.shuffle(dataset)
        x = get_X(dataset)
        y = get_Y(dataset)
        y[:] -= 1 #we need to seperate into categories, we will have to add  1 to each votes made by the model afterward
        if not is_soft_max:
            y[:] /= 4
        # seperate data into training, validation and test
        x_train = x[0:int(np.floor(x.shape[0] * 0.7))]
        y_train = y[0:int(np.floor(y.shape[0] * 0.7))]

        x_valid = x[int(np.floor(x.shape[0] * 0.7)) + 1:int(np.floor(x.shape[0] * 0.85))]
        y_valid = y[int(np.floor(y.shape[0] * 0.7)) + 1:int(np.floor(y.shape[0] * 0.85))]

        x_test = x[int(np.floor(x.shape[0] * 0.85)) + 1:int(np.floor(x.shape[0]) - 1)]
        y_test = y[int(np.floor(y.shape[0] * 0.85)) + 1:int(np.floor(y.shape[0]) - 1)]

        predictions = self.classifier.predict(x_test, batch_size=128, verbose=1)
        if is_soft_max:
            predictions = self.softmax_pred_to_list(predictions)
        else:
            predictions[:] *=4
            predictions[:] += 1
        #we run the error of the predictions, we will compare it vs mean as a baseline

        #mlp preds errors
        mlp_quad_error = utils.calculate_quadratic_error(predictions, y_test)
        mlp_abs_error = utils.calculate_abs_error(predictions, y_test)
        return mlp_abs_error, mlp_quad_error

    def generate_graph_model(self):
        from keras.utils import plot_model
        plot_model(self.classifier, to_file="model.svg", show_layer_names=True, show_shapes=True, expand_nested = True)



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
