import config
import utils
import numpy as np
import keras
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from src.classifiers.heuristic import HeuristicClassifier
from src.dataset.dataset_loader import load_covtype_dataset


app = FastAPI()

data = load_covtype_dataset()
models = {
    'heuristic': HeuristicClassifier(data),
    'decision_tree': utils.load_serialized_model(f'{config.PATH_TO_SERIALIZED_MODELS}/DecisionTree_model.pkl'),
    'random_forest': utils.load_serialized_model(f'{config.PATH_TO_SERIALIZED_MODELS}/RandomForest_model.pkl'),
    'neural_network': keras.models.load_model(f'{config.PATH_TO_SERIALIZED_MODELS}/NeuralNetwork_model.h5')
}


@app.get("/model_prediction/{model_name}")
def model_prediction(model_name: str):
    if model_name.lower() not in models.keys():
        return f'No model found. Available models are: {list(models.keys())}'

    # random row from the dataset
    input_data = data.sample()
    input_data = input_data.drop('output', axis=1).to_numpy()

    model = models.get(model_name)

    if model_name == 'heuristic':
        prediction = model.predict(input_data)
    else:
        prediction = np.argmax(model.predict(input_data)) + 1

    output = {
        'Input data': input_data.tolist(),
        'Prediction': int(prediction)
    }
    return JSONResponse(content=output)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

