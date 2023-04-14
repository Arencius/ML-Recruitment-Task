import pickle


def load_serialized_model(model_path: str):
    """
    Loads the trained and serialized model from `model_path` directory
    :param model_path: path to serialized model
    :return: serialized sklearn model
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    return model
