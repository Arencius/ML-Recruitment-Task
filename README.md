# Recruitment task for the Machine Learning Engineer role.

## Overwiew
This project aims to classify data from covertype dataset using machine learning models and a simple heuristic. The objective is to create a 
REST API that serves the best-performing model to the user, allowing them to choose between a heuristic classification, two baseline scikit-learn models, and a neural network.

## Models used in the project:

- Heuristic classification method, that classifies the input features based on the closest distance between mean value of input data and mean for each class in the dataset.
- Decision Tree Classifier: 93.74% accuracy on test set
- Random Forest Classifier: 94.96% accuracy on test set
- Neural network that contains a keras-tuner function for finding the best hyperparameters for the problem

## Models evaluation
All machine learning models evaluation plots are in the ```src/model_evaluation``` directory.
The directory contents:
- confusion matrix and ROC (Receiver Operating Characteristic curve) for DecisionTree and RandomForest classifiers
- classification report for DecisionTree and RandomForest classifiers
- training and validation accuracy of the neural network
- training and validation loss of the neural network
- neural network summary after finding the best hyperparameters

## Running the project
### System requirements:

- Python 3.7.2
- pip
### Dependencies:

1. Clone the project's source code from the repository:
```git clone https://github.com/Arencius/ML-Recruitment-Task.git```
2. Navigate to the project's directory:
```cd ML-Recruitment-Task```
3. Install all necessary dependecies:
```pip install -r requirements.txt```

### Usage:
In order to start the server, run this command in your terminal:
```uvicorn main:app --reload```. Open your web browser and navigate to http://localhost:8000/docs to view the endpoint.