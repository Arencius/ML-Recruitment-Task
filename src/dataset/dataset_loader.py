import config
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split


def load_covtype_dataset():
    """
    Loads the dataset from .csv file
    :return: None
    """
    data = pd.read_csv(config.DATASET_PATH, header=None)
    data = data.rename(columns={54: 'output'})   # for easier data manipulation (e.g. dropping the labels column)

    return data


def preprocess_data(data: pd.DataFrame):
    """
    Divides the dataset into features and labels, transforms labels from one-hot encoding,
    and splits the data into train and test set
    :param data: dataset to be preprocessed
    :return: dataset split into train and testing set
    """
    features = data.drop('output', axis=1)
    output = data['output'].values
    output = keras.utils.to_categorical(output - 1, len(set(output)))
    X_train, X_test, y_train, y_test = train_test_split(features, output, test_size=config.DATASET_TEST_SIZE)

    return X_train, X_test, y_train, y_test


