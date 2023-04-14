import os
import config
import matplotlib.pyplot as plt
import tensorflow as tf
import keras_tuner
import keras
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras_tuner.tuners import RandomSearch
from src.dataset.dataset_loader import load_covtype_dataset, preprocess_data


def build_model(hp: keras_tuner.HyperParameters) -> keras.Sequential:
    """
    Creates and compiles a Keras Sequential model based on given hyperparameters
    :param hp: hyperparameters to be optimized during the model tuning process
    :return: keras Sequential model
    """
    model = keras.Sequential()

    for i in range(hp.Int("num_layers", 1, 4)):
        model.add(Dense(units=hp.Int(f"units_{i}", min_value=32, max_value=128, step=32),
                        activation='relu'))

    if hp.Boolean("dropout"):
        model.add(Dropout(rate=0.3))

    model.add(Dense(units=7, activation='softmax'))

    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


class NeuralNetwork:
    def __init__(self):
        """
        Neural network class, that handles the operations of finding the best parameters using keras-tuner, training,
        plotting the results (accuracy and loss) and serializing the trained network.
        """
        self.name = 'NeuralNetwork'
        self.best_model = None
        self.dataset = load_covtype_dataset()
        self.X_train, self.X_test, self.y_train, self.y_test = preprocess_data(self.dataset)

        self.model_tuner = RandomSearch(
            hypermodel=build_model,
            objective="val_accuracy",
            max_trials=2,
            executions_per_trial=2,
            overwrite=True
        )

    def _plot_training_results(self,
                               history: tf.keras.callbacks.History,
                               mode: str = 'accuracy') -> None:
        """
        Enables visualisation of the training results
        :param history: training history of the model
        :param mode: str, one of [accuracy, loss] - whether to plot accuracy or loss history
        :return: None
        """
        plt.clf()  # clears plot
        plt.plot(history.history[mode])
        plt.plot(history.history[f'val_{mode}'])
        plt.title(f'Model {mode}')
        plt.ylabel(mode)
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.grid()

        path = os.path.join(config.PATH_TO_EVALUATION, f'{self.name}_{mode}.png')
        plt.savefig(path)

    def find_best_model(self) -> keras.models.Sequential:
        """
        Based on given hyperparameters tunes the model, to find one with best validation accuracy for given problem
        :return: best sequential model after hyperparameter tuning
        """
        self.model_tuner.search(self.X_train, self.y_train,
                                epochs=5,
                                validation_data=(self.X_test, self.y_test),
                                verbose=1)
        return self.model_tuner.get_best_models()[0]

    def train_model(self) -> None:
        """
        After finding the best hyperparameters for the model, trains it on the dataset,
        plots the results and serializes the model.
        :return: None
        """
        self.best_model = self.find_best_model()
        history = self.best_model.fit(self.X_train, self.y_train,
                                      epochs=10,
                                      validation_data=(self.X_test, self.y_test))

        self._plot_training_results(history, mode='accuracy')
        self._plot_training_results(history, mode='loss')

        path = os.path.join(config.PATH_TO_SERIALIZED_MODELS, f'{self.name}_model.h5')
        self.best_model.save(path)
