import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from abc import ABC
from sklearn.metrics import confusion_matrix, roc_curve, auc
import config
from src.dataset.dataset_loader import load_covtype_dataset, preprocess_data


class BaseClassifier(ABC):
    def __init__(self):
        """
        Abstract class for scikit-learn classifiers. Implements logic for model evaluation, training and serializing the model
        """
        self.classifier = None
        self.name = None
        self.dataset = load_covtype_dataset()
        self.X_train, self.X_test, self.y_train, self.y_test = preprocess_data(self.dataset)

    def _plot_confusion_matrix(self, y_pred: np.array,
                               y_test: np.array) -> None:
        cm = confusion_matrix(y_test, y_pred, labels=np.arange(7))

        fig, ax = plt.subplots()
        ax.matshow(cm, cmap=plt.cm.Blues)

        ax.set_xticklabels([''] + list(np.arange(7)))
        ax.set_yticklabels([''] + list(np.arange(7)))

        for i in range(len(cm)):
            for j in range(len(cm)):
                ax.text(j, i, str(cm[i][j]), ha="center", va="center")

        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

        path = os.path.join(config.PATH_TO_EVALUATION, f'{self.name}_confusion_matrix.png')
        plt.savefig(path)

    def _plot_roc_curve(self, y_pred: np.array,
                        y_test: np.array) -> None:
        """
        Visualizes the Receiver Operating Characteristic curve (ROC)
        :param y_pred: model predictions
        :param y_test: ground truth labels from test dataset
        :return: None
        """
        fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='orange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic curve')
        plt.legend(loc="lower right")

        path = os.path.join(config.PATH_TO_EVALUATION, f'{self.name}_roc_curve.png')
        plt.savefig(path)

    def _one_hot_encoding_to_labels(self, encoded_labels: np.array) -> np.array:
        """
        Transforms the labels from one-hot encoded to numerical
        :param encoded_labels: one-hot encoded labels
        :return: transformed labels
        """
        return np.argmax(encoded_labels, axis=1) + 1

    def _serialize_model(self) -> None:
        """
        Saves the trained model in `serialized_models` directory
        :return: None
        """
        serialized_model_path = os.path.join(config.PATH_TO_SERIALIZED_MODELS, f'{self.name}_model.pkl')
        with open(serialized_model_path, 'wb') as file:
            pickle.dump(self.classifier, file)

    def train_and_save_model(self):
        """
        Trains the classifier, plotting the evaluation metrics and then serializes it in `serialized_models` directory
        :return:
        """
        self.classifier.fit(self.X_train, self.y_train)

        predictions = self.classifier.predict(self.X_test)
        predictions = self._one_hot_encoding_to_labels(predictions) - 1
        y_test = self._one_hot_encoding_to_labels(self.y_test) - 1

        self._plot_confusion_matrix(predictions, y_test)
        self._plot_roc_curve(predictions, y_test)

        self._serialize_model()
