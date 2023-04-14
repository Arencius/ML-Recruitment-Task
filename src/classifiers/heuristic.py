import numpy as np


class HeuristicClassifier:
    def __init__(self, dataset):
        """
        Data classification using heuristic method.
        For each class in dataset the average value across all columns is computed, and then compared to the mean of the input data.
        The output is an index of the class with closest 'distance' between class mean and the input mean value.
        """
        self.dataset = dataset

    def _compute_classes_averages(self) -> dict:
        """
        Computes average values across all columns for each class
        :return: dictionary in the form {class_index: average value from all dataset instances for given class (float)}
        """
        output_classes_features = self.dataset.groupby('output').mean()  # average values in each column for every class
        class_averages = dict(output_classes_features.mean(axis=1))  # average value across all columns for every class

        return class_averages

    def predict(self, input_data: np.array) -> int:
        """
        Predicts the corresponding class of the input, based on distance between class mean value and input mean value
        :param input_data: one row of the input data
        :return: index of the closest class for given input
        """
        input_data_mean = input_data.mean()  # average of all input features
        class_averages = self._compute_classes_averages()

        closest_class_index = min(class_averages, key=lambda k: abs(class_averages[k] - input_data_mean))
        return closest_class_index - 1
