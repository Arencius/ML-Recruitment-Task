import os

DATASET_TEST_SIZE = 0.25
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(ROOT_DIR, 'src/dataset/covtype.csv')
PATH_TO_EVALUATION = os.path.join(ROOT_DIR, 'src/model_evaluation/')
PATH_TO_SERIALIZED_MODELS = os.path.join(ROOT_DIR, 'src/serialized_models/')
