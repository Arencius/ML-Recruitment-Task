from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from src.classifiers.base_classifier import BaseClassifier


class DecisionTree(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.classifier = DecisionTreeClassifier()
        self.name = 'DecisionTree'


class RandomForest(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.classifier = RandomForestClassifier(n_estimators=125)
        self.name = 'RandomForest'
